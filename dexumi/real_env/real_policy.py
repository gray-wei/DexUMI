import os

import cv2
import numpy as np
import torch
from dexumi.common.utility.file import read_pickle
from dexumi.common.utility.model import load_config, load_diffusion_model
from dexumi.constants import INPAINT_RESIZE_RATIO
from dexumi.diffusion_policy.dataloader.diffusion_bc_dataset import (
    normalize_data,
    process_image,
    unnormalize_data,
)


class RealPolicy:
    def __init__(
        self,
        model_path: str,
        ckpt: int,
    ):
        model_cfg = load_config(model_path)
        model, noise_scheduler = load_diffusion_model(
            model_path, ckpt, use_ema=model_cfg.training.use_ema
        )
        stats = read_pickle(os.path.join(model_path, "stats.pickle"))
        self.pred_horizon = model_cfg.dataset.pred_horizon
        self.action_dim = model_cfg.action_dim
        self.obs_horizon = model_cfg.dataset.obs_horizon

        self.model = model.eval()
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = model_cfg.num_inference_steps
        self.stats = stats
        self.camera_resize_shape = model_cfg.dataset.camera_resize_shape
        if model_cfg.dataset.relative_hand_action:
            print("Using relative hand action")
            print("hand_action stats", stats["relative_hand_action"])
            print(
                stats["relative_hand_action"]["max"]
                - stats["relative_hand_action"]["min"]
                > 5e-2
            )
            self.stats["action"] = {
                "min": np.concatenate(
                    [
                        stats["relative_pose"]["min"],
                        stats["relative_hand_action"]["min"],
                    ]
                ),
                "max": np.concatenate(
                    [
                        stats["relative_pose"]["max"],
                        stats["relative_hand_action"]["max"],
                    ]
                ),
            }
        else:
            print("Using absolute hand action")
            print("hand_action stats", stats["hand_action"])
            print(stats["hand_action"]["max"] - stats["hand_action"]["min"] > 5e-2)
            self.stats["action"] = {
                "min": np.concatenate(
                    [stats["relative_pose"]["min"], stats["hand_action"]["min"]]
                ),
                "max": np.concatenate(
                    [stats["relative_pose"]["max"], stats["hand_action"]["max"]]
                ),
            }
        print(self.stats["action"])
        self.model_cfg = model_cfg

    def predict_action(self, proprioception, fsr, visual_obs):
        # visual_obs: NxHxWxC
        _, H, W, _ = visual_obs.shape
        B = 1
        if proprioception is not None and "proprioception" in self.stats:
            proprioception = normalize_data(
                proprioception.reshape(1, -1), self.stats["proprioception"]
            )  # (1,N)
            proprioception = (
                torch.from_numpy(proprioception).unsqueeze(0).cuda()
            )  # (B,1,6)
        elif proprioception is not None:
            print("Warning: proprioception data provided but no stats available, setting to None")
            proprioception = None

        if fsr is not None and "fsr" in self.stats:
            fsr = normalize_data(fsr.reshape(1, -1), self.stats["fsr"])
            fsr = torch.from_numpy(fsr).unsqueeze(0).cuda()  # (B,1,2)
        elif fsr is not None:
            print("Warning: fsr data provided but no stats available, setting to None")
            fsr = None

        visual_obs = np.array(
            [
                cv2.cvtColor(
                    cv2.resize(
                        obs,
                        (
                            int(W * INPAINT_RESIZE_RATIO),
                            int(H * INPAINT_RESIZE_RATIO),
                        ),
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                for obs in visual_obs
            ]
        )
        visual_obs = process_image(
            visual_obs,
            optional_transforms=["Resize", "CenterCrop"],
            resize_shape=self.camera_resize_shape,
        )
        visual_obs = visual_obs.unsqueeze(0).cuda()
        trajectory = torch.randn(B, self.pred_horizon, self.action_dim).cuda()
        trajectory = self.model.inference(
            proprioception=proprioception,
            fsr=fsr,
            visual_obs=visual_obs,
            trajectory=trajectory,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.num_inference_steps,
        )
        trajectory = trajectory.detach().to("cpu").numpy()
        naction = trajectory[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])
        start = self.obs_horizon - 1
        end = start + self.pred_horizon
        action = action_pred[start:end, :]
        return action
