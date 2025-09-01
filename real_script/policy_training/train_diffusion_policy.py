import copy
import datetime
import os
import pickle

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from dexumi.common.utility.model import load_config
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="../../config/diffusion_policy",
    config_name="train_diffusion_policy",
)
def train_diffusion_policy(cfg: DictConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    # # create save dir
    output_dir = HydraConfig.get().runtime.output_dir
    resume = cfg.training.resume
    if resume:
        # overwrite the config with the checkpoint config
        accelerator.print("resume from checkpoint")
        model_path = cfg.training.model_path
        model_ckpt = cfg.training.model_ckpt
        cfg = load_config(model_path)
        # save the checkpoint config to the output dir
        OmegaConf.save(cfg, os.path.join(output_dir, ".hydra", "config.yaml"))
    if accelerator.is_local_main_process:
        wandb.init(project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))
        accelerator.print("Logging dir", output_dir)
        ckpt_save_dir = os.path.join(output_dir, "checkpoints")
        state_save_dir = os.path.join(output_dir, "state")
        os.makedirs(ckpt_save_dir, exist_ok=True)
        os.makedirs(state_save_dir, exist_ok=True)

    # max_episode = 5 if cfg.debug else cfg.dataset.max_episode
    dataset = hydra.utils.instantiate(
        cfg.dataset, max_episode=5 if cfg.debug else cfg.dataset.max_episode
    )
    print("Total training samples:", len(dataset))
    # open a file for writing in binary mode
    if accelerator.is_local_main_process:
        # save training data statistics (min, max) for each dim
        print("saving stats...")
        stats = dataset.stats
        print("checking if normalize...")
        if cfg.dataset.relative_hand_action:
            print(
                stats["relative_hand_action"]["max"]
                - stats["relative_hand_action"]["min"]
                > 5e-2
            )
        else:
            print(stats["hand_action"]["max"] - stats["hand_action"]["min"] > 5e-2)
        with open(os.path.join(output_dir, "stats.pickle"), "wb") as f:
            # write the dictionary to the file
            pickle.dump(stats, f)
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.training.shuffle,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        drop_last=cfg.training.drop_last,
    )
    print("====================================")
    print("len of dataset", len(dataset))
    print("====================================")

    sample_batch = next(iter(dataloader))
    for k, v in sample_batch.items():
        accelerator.print(k, v.shape)

    model = hydra.utils.instantiate(cfg.model)
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = cfg.training.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    print("Preparing model, optimizer, dataloader, lr_scheduler...")
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    print("✓ Accelerator preparation completed")
    if resume:
        model_state = os.path.join(model_path, "state", f"epoch_{model_ckpt}")
        accelerator.load_state(model_state)
        print("successfully loaded model from checkpoint!")

    if cfg.training.use_ema:
        print("Creating EMA model...")
        ema_model = copy.deepcopy(accelerator.unwrap_model(model))
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
        print("✓ EMA model created")

    print("🚀 Starting training loop...")
    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch}/{cfg.training.epochs}")
        epoch_loss = []

        # batch loop
        for batch in dataloader:
            with accelerator.accumulate(model):
                (visual_observation, actions, fsr, proprioception) = (
                    batch["camera_0"].to(accelerator.device),
                    batch["action"].to(accelerator.device),
                    batch["fsr"].to(accelerator.device) if "fsr" in batch else None,
                    batch["proprioception"].to(accelerator.device)
                    if "proprioception" in batch
                    else None,
                )
                noise = torch.randn(actions.shape, device=accelerator.device)
                bsz = actions.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                ).long()
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                loss = model(
                    noisy_actions=noisy_actions,
                    timesteps=timesteps,
                    proprioception=proprioception,
                    fsr=fsr,
                    visual_obs=visual_observation,
                    noise=noise,
                )

                # optimize
                optimizer.zero_grad()
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                # update ema
                if cfg.training.use_ema:
                    ema.step(accelerator.unwrap_model(model))
                # logging
                epoch_loss.append(loss.item())
        if accelerator.is_local_main_process and not cfg.debug:
            wandb.log(
                {
                    "epoch loss": np.mean(epoch_loss),
                }
            )
        if epoch % cfg.training.ckpt_frequency == 0 and epoch > 0:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                ckpt_model = accelerator.unwrap_model(model)
                accelerator.save(
                    ckpt_model.state_dict(),
                    os.path.join(ckpt_save_dir, f"epoch_{epoch}.ckpt"),
                )
                accelerator.print(f"Saved checkpoint at epoch {epoch}.")
                if cfg.training.use_ema:
                    accelerator.save(
                        ema_model.state_dict(),
                        os.path.join(ckpt_save_dir, f"ema_epoch_{epoch}.ckpt"),
                    )
                    accelerator.print(f"Saved ema checkpoint at epoch {epoch}.")
                # also save the state
                accelerator.save_state(
                    output_dir=os.path.join(state_save_dir, f"epoch_{epoch}")
                )
                accelerator.print(f"Saved state checkpoint at epoch {epoch}.")


if __name__ == "__main__":
    train_diffusion_policy()
