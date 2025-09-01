import timm
import torch
from dexumi.common.utility.model import freeze_model
from einops import rearrange
from torch import nn


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        vision_backbone_kwargs,
        freeze_vision_back,
        diffusion_policy_head,
    ):
        super().__init__()
        self.vision_backbone = timm.create_model(
            **vision_backbone_kwargs, pretrained=True
        )
        self.freeze_vision_back = freeze_vision_back
        if self.freeze_vision_back:
            freeze_model(self.vision_backbone)
        self.vision_backbone.train()
        self.diffusion_policy_head = diffusion_policy_head

    def forward(
        self,
        noisy_actions,
        timesteps,
        proprioception,
        fsr,
        visual_obs,
        noise,
    ):
        bsz, obs_horizon, C, H, W = visual_obs.shape
        visual_obs = rearrange(visual_obs, "b o c h w -> (b o) c h w")
        visual_embedding = self.vision_backbone(visual_obs)  # (bsz * obs_horizon, D)
        visual_embedding = rearrange(
            visual_embedding, "(b o) d -> b (o d)", b=bsz, o=obs_horizon
        )
        condition = [visual_embedding]
        if proprioception is not None:
            condition.append(proprioception.flatten(start_dim=1))
        if fsr is not None:
            condition.append(fsr.flatten(start_dim=1))

        condition = torch.cat(condition, dim=1)
        noise_prediction = self.diffusion_policy_head(
            noisy_actions, timesteps, global_cond=condition
        )
        # diffusion loss
        loss = nn.functional.mse_loss(noise_prediction, noise)
        return loss

    def condition_sample(self, cond, trajectory, noise_scheduler):
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                model_output = self.diffusion_policy_head(
                    trajectory, t.unsqueeze(0).cuda(), cond
                )
            trajectory = noise_scheduler.step(model_output, t, trajectory).prev_sample

        return trajectory

    def inference(
        self,
        proprioception,
        fsr,
        visual_obs,
        trajectory,
        noise_scheduler,
        num_inference_steps,
    ):
        noise_scheduler.set_timesteps(num_inference_steps)
        bsz, obs_horizon, C, H, W = visual_obs.shape
        visual_obs = rearrange(visual_obs, "b o c h w -> (b o) c h w")
        visual_embedding = self.vision_backbone(visual_obs)  # (bsz * obs_horizon, D)
        visual_embedding = rearrange(
            visual_embedding, "(b o) d -> b (o d)", b=bsz, o=obs_horizon
        )
        condition = [visual_embedding]
        print(f"    visual_embedding shape: {visual_embedding.shape}")
        if proprioception is not None:
            proprioception = rearrange(proprioception, "b o c -> b (o c)")
            condition.append(proprioception)
            print(f"    proprioception shape: {proprioception.shape}")
        if fsr is not None:
            fsr = rearrange(fsr, "b o c -> b (o c)")
            condition.append(fsr)
            print(f"    fsr shape: {fsr.shape}")

        condition = torch.cat(condition, dim=1)
        print(f"    Final condition shape: {condition.shape}")
        trajectory = self.condition_sample(condition, trajectory, noise_scheduler)
        return trajectory
