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
    """
    真实环境策略类 - 用于在真实机器人环境中执行动作预测
    基于扩散模型的机器人策略，可以处理多模态输入（视觉、本体感受、力传感器数据）
    """
    def __init__(
        self,
        model_path: str,  # 模型路径
        ckpt: int,        # 检查点编号
    ):
        """
        初始化真实环境策略
        
        Args:
            model_path: 训练好的模型文件路径
            ckpt: 要加载的检查点编号
        """
        # 加载模型配置文件
        model_cfg = load_config(model_path)
        
        # 加载扩散模型和噪声调度器
        model, noise_scheduler = load_diffusion_model(
            model_path, ckpt, use_ema=model_cfg.training.use_ema
        )
        
        # 加载数据统计信息（用于数据归一化和反归一化）
        stats = read_pickle(os.path.join(model_path, "stats.pickle"))
        
        # 设置模型参数
        self.pred_horizon = model_cfg.dataset.pred_horizon      # 预测时间范围
        self.action_dim = model_cfg.action_dim                  # 动作维度
        self.obs_horizon = model_cfg.dataset.obs_horizon       # 观测时间范围

        # 将模型设置为评估模式
        self.model = model.eval()
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = model_cfg.num_inference_steps  # 推理步数
        self.stats = stats
        self.camera_resize_shape = model_cfg.dataset.camera_resize_shape  # 相机图像调整尺寸
        # 处理手部动作类型（相对位置 vs 绝对位置）
        if model_cfg.dataset.relative_hand_action:
            print("Using relative hand action")  # 使用相对手部动作
            print("hand_action stats", stats["relative_hand_action"])
            print(
                stats["relative_hand_action"]["max"]
                - stats["relative_hand_action"]["min"]
                > 5e-2
            )
            # 组合相对姿态和相对手部动作的统计信息
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
            print("Using absolute hand action")  # 使用绝对手部动作
            print("hand_action stats", stats["hand_action"])
            print(stats["hand_action"]["max"] - stats["hand_action"]["min"] > 5e-2)
            # 组合相对姿态和绝对手部动作的统计信息
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
        """
        预测机器人动作
        
        Args:
            proprioception: 本体感受数据（关节位置、速度等）
            fsr: 力传感器数据
            visual_obs: 视觉观测数据，形状为 NxHxWxC
            
        Returns:
            action: 预测的动作序列
        """
        # 获取视觉观测的尺寸信息
        _, H, W, _ = visual_obs.shape
        B = 1  # 批次大小为1（单次预测）
        # 处理本体感受数据
        if proprioception is not None and "proprioception" in self.stats:
            # 归一化本体感受数据
            proprioception = normalize_data(
                proprioception.reshape(1, -1), self.stats["proprioception"]
            )  # (1,N)
            # 转换为PyTorch张量并移到GPU
            proprioception = (
                torch.from_numpy(proprioception).unsqueeze(0).cuda()
            )  # (B,1,6)
        elif proprioception is not None:
            print("Warning: proprioception data provided but no stats available, setting to None")
            proprioception = None

        # 处理力传感器数据
        if fsr is not None and "fsr" in self.stats:
            # 归一化力传感器数据
            fsr = normalize_data(fsr.reshape(1, -1), self.stats["fsr"])
            fsr = torch.from_numpy(fsr).unsqueeze(0).cuda()  # (B,1,2)
        elif fsr is not None:
            print("Warning: fsr data provided but no stats available, setting to None")
            fsr = None

        # 处理视觉观测数据
        # 1. 调整图像尺寸并转换颜色空间（BGR到RGB）
        visual_obs = np.array(
            [
                cv2.cvtColor(
                    cv2.resize(
                        obs,
                        (
                            int(W * INPAINT_RESIZE_RATIO),  # 按比例调整宽度
                            int(H * INPAINT_RESIZE_RATIO),  # 按比例调整高度
                        ),
                    ),
                    cv2.COLOR_BGR2RGB,  # BGR转RGB
                )
                for obs in visual_obs
            ]
        )
        
        # 2. 进一步处理图像（调整大小、中心裁剪等）
        visual_obs = process_image(
            visual_obs,
            optional_transforms=["Resize", "CenterCrop"],
            resize_shape=self.camera_resize_shape,
        )
        
        # 3. 转换为PyTorch张量并移到GPU
        visual_obs = visual_obs.unsqueeze(0).cuda()
        # 初始化随机轨迹作为扩散模型的起点
        trajectory = torch.randn(B, self.pred_horizon, self.action_dim).cuda()
        
        # 使用扩散模型进行推理，生成动作轨迹
        trajectory = self.model.inference(
            proprioception=proprioception,
            fsr=fsr,
            visual_obs=visual_obs,
            trajectory=trajectory,
            noise_scheduler=self.noise_scheduler,
            num_inference_steps=self.num_inference_steps,
        )
        
        # 将结果转移到CPU并转换为numpy数组
        trajectory = trajectory.detach().to("cpu").numpy()
        naction = trajectory[0]  # 获取第一个批次的结果
        
        # 反归一化动作数据，恢复到原始尺度
        action_pred = unnormalize_data(naction, stats=self.stats["action"])
        
        # 提取有效的动作序列（从观测范围结束到预测范围结束）
        start = self.obs_horizon - 1
        end = start + self.pred_horizon
        action = action_pred[start:end, :]
        
        return action
