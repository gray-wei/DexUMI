#!/usr/bin/env python3
"""
离线策略验证脚本 - 用于调试模型预测问题
通过加载训练数据集进行单步预测验证，对比预测结果与ground truth

配置: 仅使用Vision模态 (不使用proprioception和FSR)
适用于纯视觉模型的调试和验证
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import zarr
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dexumi.real_env.real_policy import RealPolicy
from dexumi.diffusion_policy.dataloader.dexumi_dataset import DexUMIDataset
from dexumi.diffusion_policy.dataloader.diffusion_bc_dataset import process_image
from dexumi.common.utility.matrix import (
    homogeneous_matrix_to_6dof,
    vec6dof_to_homogeneous_matrix,
    relative_transformation,
    invert_transformation,
)


class OfflinePolicyDebugger:
    """离线策略调试器"""
    
    def __init__(self, model_path, ckpt, dataset_path):
        """
        初始化调试器
        
        Args:
            model_path: 模型路径
            ckpt: 检查点编号
            dataset_path: 数据集路径
        """
        print("🔧 初始化调试器...")
        
        # 加载策略模型
        self.policy = RealPolicy(model_path=model_path, ckpt=ckpt)
        print(f"✅ 模型已加载: {model_path}/checkpoint_{ckpt}.pth")
        
        # 加载数据集
        self.dataset_path = dataset_path
        self.load_dataset()
        
    def load_dataset(self):
        """
        加载完整数据集用于离线验证
        """
        print(f"📂 加载数据集: {self.dataset_path}")
        
        # 检查数据集路径是否存在
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        
        # 使用与训练完全相同的配置
        config = self.policy.model_cfg.dataset
        
        # 自动检测可用的camera IDs
        print(f"    🔍 检测可用的相机配置...")
        available_camera_ids = self._detect_available_camera_ids()
        
        # 使用配置文件中的camera_ids，如果数据集中不存在则使用第一个可用的
        config_camera_ids = getattr(config, 'load_camera_ids', [0])
        camera_ids = []
        for cam_id in config_camera_ids:
            if cam_id in available_camera_ids:
                camera_ids.append(cam_id)
        
        # 如果配置的camera_ids都不可用，使用第一个可用的
        if not camera_ids and available_camera_ids:
            camera_ids = [available_camera_ids[0]]
            print(f"    ⚠️ 配置的camera_ids {config_camera_ids} 不可用，使用 {camera_ids}")
        
        print(f"    📷 使用相机ID: {camera_ids}")
        print(f"    📋 训练配置参数:")
        print(f"      skip_proprioception: {config.skip_proprioception}")
        print(f"      enable_fsr: {config.enable_fsr}")
        print(f"      bgr2rgb: {config.bgr2rgb}")
        print(f"      relative_hand_action: {config.relative_hand_action}")
        
        # 对于离线验证，禁用随机变换以确保确定性结果
        deterministic_transforms = ["Resize", "CenterCrop"] if hasattr(config, 'optional_transforms') and config.optional_transforms else []
        print(f"    🎲 transforms: {deterministic_transforms} (移除随机变换确保确定性)")
        
        self.dataset = DexUMIDataset(
            data_dirs=[self.dataset_path],
            max_episode=None,
            load_camera_ids=camera_ids,
            camera_resize_shape=config.camera_resize_shape,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            unnormal_list=config.unnormal_list,
            relative_hand_action=config.relative_hand_action,
            skip_proprioception=config.skip_proprioception,  # 使用训练配置
            enable_fsr=config.enable_fsr,                    # 使用训练配置
            bgr2rgb=config.bgr2rgb,
            optional_transforms=deterministic_transforms,    # 确定性变换
            fsr_binary_cutoff=config.fsr_binary_cutoff,      # FSR二值化阈值
        )
        
        print(f"✅ 数据集已加载，共 {len(self.dataset)} 个样本")
        
    def _detect_available_camera_ids(self):
        """
        检测数据集中可用的camera IDs
        
        Returns:
            List[int]: 可用的camera IDs
        """
        available_camera_ids = []
        try:
            root = zarr.open(self.dataset_path, mode="r")
            episodes = sorted(list(root.group_keys()))
            
            if episodes:
                # 检查第一个episode中的camera keys
                first_episode = root[episodes[0]]
                for key in first_episode.keys():
                    if key.startswith('camera_'):
                        try:
                            cam_id = int(key.split('_')[1])
                            available_camera_ids.append(cam_id)
                        except (IndexError, ValueError):
                            continue
                            
                available_camera_ids.sort()
                print(f"      检测到相机IDs: {available_camera_ids}")
                
        except Exception as e:
            print(f"      ⚠️ 检测相机IDs失败: {e}")
            
        return available_camera_ids
        
    def load_raw_episode_data(self, episode_idx=0):
        """
        直接从zarr文件加载原始episode数据
        
        Args:
            episode_idx: episode索引
            
        Returns:
            dict: 包含原始数据的字典
        """
        print(f"📖 加载原始episode数据: episode_{episode_idx}")
        
        root = zarr.open(self.dataset_path, mode="r")
        episodes = sorted(list(root.group_keys()))
        
        if episode_idx >= len(episodes):
            raise ValueError(f"Episode {episode_idx} 不存在，总共只有 {len(episodes)} 个episodes")
            
        episode_name = episodes[episode_idx]
        episode_group = root[episode_name]
        
        # 加载所有数据
        data = {}
        for key in episode_group.keys():
            if key.startswith('camera_'):
                # Camera数据是group，包含rgb子数组
                camera_group = episode_group[key]
                if 'rgb' in camera_group:
                    data[key] = {'rgb': np.array(camera_group['rgb'])}
                    print(f"  {key}: group with rgb shape {data[key]['rgb'].shape}")
                else:
                    # 向后兼容：如果不是group结构
                    data[key] = np.array(episode_group[key])
                    print(f"  {key}: {data[key].shape}")
            else:
                data[key] = np.array(episode_group[key])
                print(f"  {key}: {data[key].shape}")
            
        print(f"✅ Episode {episode_name} 数据已加载")
        return data, episode_name
        
    def _prepare_visual_input(self, raw_image):
        """
        准备视觉输入，复用real_policy.py的处理逻辑
        
        Args:
            raw_image: 原始图像，应该是BGR格式 (H, W, 3)
            
        Returns:
            np.ndarray: 处理后的BGR图像，用于传递给policy.predict_action
        """
        print(f"🖼️ 准备视觉输入")
        print(f"  输入图像: {raw_image.shape}, 格式: BGR")
        
        # 数据集中的图像已经是裁剪后的193×238 BGR格式
        # 直接返回给real_policy.py处理，它会内部进行BGR→RGB转换和resize
        return raw_image
        
    def single_step_prediction(self, episode_idx=0, step_idx=0):
        """
        进行单步预测并与ground truth对比
        
        Args:
            episode_idx: episode索引
            step_idx: 时间步索引
            
        Returns:
            dict: 预测结果和对比信息
        """
        print(f"\n🎯 执行单步预测验证 (episode {episode_idx}, step {step_idx})")
        
        # 1. 加载原始数据
        raw_data, episode_name = self.load_raw_episode_data(episode_idx)
        
        # 2. 获取指定步骤的数据
        # 自动检测camera key
        camera_keys = [key for key in raw_data.keys() if key.startswith('camera_')]
        if not camera_keys:
            raise ValueError("未找到相机数据")
        
        camera_key = camera_keys[0]
        
        # 获取初始绝对pose用于轨迹重建
        initial_absolute_pose = raw_data['pose'][step_idx]  # 第一帧的绝对pose
        print(f"  🌍 初始绝对pose: {initial_absolute_pose}")
        
        # 检查步骤索引是否有效 - 注意camera数据是group，需要检查rgb数据
        if 'rgb' in raw_data[camera_key]:
            rgb_data = raw_data[camera_key]['rgb']
            max_step = len(rgb_data) - 1
            
            # 处理边界情况
            if step_idx >= len(rgb_data):
                print(f"    ⚠️ Step {step_idx} 超出范围(最大{max_step})，调整为最大可用步骤")
                step_idx = max_step
                
            # 检查是否有足够的后续步骤用于16步预测
            remaining_steps = len(rgb_data) - step_idx
            if remaining_steps < 16:
                print(f"    ⚠️ 从step {step_idx}开始只有{remaining_steps}步数据，不足16步")
                # 调整起始位置以确保有16步数据
                if len(rgb_data) >= 16:
                    step_idx = len(rgb_data) - 16
                    print(f"    🔧 调整起始位置为step {step_idx}以确保16步数据")
                else:
                    print(f"    ❌ Episode总长度{len(rgb_data)}不足16步，无法进行完整测试")
            
            raw_image = rgb_data[step_idx]  # (H, W, 3)
        else:
            # 如果不是group结构，直接访问
            max_step = len(raw_data[camera_key]) - 1
            
            if step_idx >= len(raw_data[camera_key]):
                print(f"    ⚠️ Step {step_idx} 超出范围(最大{max_step})，调整为最大可用步骤")
                step_idx = max_step
                
            remaining_steps = len(raw_data[camera_key]) - step_idx
            if remaining_steps < 16:
                print(f"    ⚠️ 从step {step_idx}开始只有{remaining_steps}步数据，不足16步")
                if len(raw_data[camera_key]) >= 16:
                    step_idx = len(raw_data[camera_key]) - 16
                    print(f"    🔧 调整起始位置为step {step_idx}以确保16步数据")
                    
            raw_image = raw_data[camera_key][step_idx]  # (H, W, 3)
            
        raw_pose = raw_data['pose'][step_idx]       # 可能是6D或7D
        
        print(f"  📋 原始数据:")
        print(f"    图像: {raw_image.shape} (来自 {camera_key})")
        print(f"    姿态: {raw_pose.shape} = {raw_pose}")
        print(f"    本体感受: {raw_data['proprioception'][step_idx].shape}")
        print(f"    FSR: {raw_data['fsr'][step_idx].shape} = {raw_data['fsr'][step_idx]}")
            
        # 3. 准备视觉输入
        visual_input = self._prepare_visual_input(raw_image)
        
        # 4. 获取ground truth action - 完整16步序列
        # 从训练数据集中获取处理后的action序列（relative + normalized）
        try:
            target_sample_idx = self.find_matching_sample_index(episode_idx, step_idx)
            
            dataset_sample = self.dataset[target_sample_idx]
            
            # 获取完整的16步action序列
            gt_actions = dataset_sample['action']  # shape: (pred_horizon, action_dim) = (16, 18)
            
            print(f"    📊 使用数据集样本索引: {target_sample_idx}")
            print(f"    📐 GT Action序列形状: {gt_actions.shape}")
            
            # 验证数据合理性
            if len(gt_actions.shape) != 2 or gt_actions.shape[0] != 16 or gt_actions.shape[1] != 18:
                raise ValueError(f"GT action形状异常: {gt_actions.shape}, 期望: (16, 18)")
                
        except (IndexError, KeyError, ValueError) as e:
            print(f"    ⚠️ 获取GT数据失败: {e}")
            print(f"    🔄 回退到第一个样本作为参考")
            dataset_sample = self.dataset[0]
            gt_actions = dataset_sample['action']
        
        print(f"  🎯 Ground Truth Actions: {gt_actions.shape}")
        print(f"    第1步: {gt_actions[0]}")
        print(f"    第8步: {gt_actions[7]}")  
        print(f"    第16步: {gt_actions[15]}")
        
        # 5. 模型预测 - 使用真实的模态数据
        print("  🤖 执行模型预测...")
        
        # 准备模态输入数据
        proprioception_input = None
        fsr_input = None
        
        # 根据模型配置决定是否使用proprioception
        if not self.policy.model_cfg.dataset.skip_proprioception:
            raw_proprioception = raw_data['proprioception'][step_idx]  # (14,)
            proprioception_input = raw_proprioception[None, ...]  # 添加批次维度
            print(f"    Proprioception: {proprioception_input.shape}")
        else:
            print("    Proprioception: 跳过")
        
        # 根据模型配置决定是否使用FSR
        if self.policy.model_cfg.dataset.enable_fsr:
            raw_fsr = raw_data['fsr'][step_idx]  # (3,)
            # 应用与训练时相同的binary cutoff
            binary_cutoff = getattr(self.policy.model_cfg.dataset, 'fsr_binary_cutoff', [10, 10, 10])
            fsr_binary = (raw_fsr >= binary_cutoff).astype(np.float32)
            fsr_input = fsr_binary.reshape(1, -1)  # 添加批次维度
            print(f"    FSR: {fsr_input.shape}, raw: {raw_fsr}, binary: {fsr_input[0]}")
        else:
            print("    FSR: 禁用")
        
        # 视觉输入 - BGR格式，添加批次维度
        visual_input_batch = visual_input[None, ...]
        print(f"    Visual: {visual_input_batch.shape}, 格式: BGR")
        
        # 执行预测 - 直接调用RealPolicy的predict_action方法
        print("  🔄 开始模型推理...")
        try:
            import torch
            with torch.no_grad():  # 禁用梯度计算以节省内存
                predicted_action = self.policy.predict_action(
                    proprioception=proprioception_input,
                    fsr=fsr_input,
                    visual_obs=visual_input_batch
                )
            print("  ✅ 模型推理完成")
        except Exception as e:
            print(f"  ❌ 模型推理失败: {e}")
            raise
        
        print(f"  🔮 预测Actions: {predicted_action.shape}")
        print(f"    第1步: {predicted_action[0]}")
        print(f"    第8步: {predicted_action[7]}")
        print(f"    第16步: {predicted_action[15]}")
        
        # 6. 完整16步对比分析
        print(f"\n📊 执行16步序列对比分析...")
        
        # 确保数据格式一致
        if predicted_action.shape != gt_actions.shape:
            print(f"    ⚠️ 形状不匹配: 预测{predicted_action.shape} vs GT{gt_actions.shape}")
            # 尝试调整到相同形状
            min_steps = min(predicted_action.shape[0], gt_actions.shape[0])
            predicted_action = predicted_action[:min_steps]
            gt_actions = gt_actions[:min_steps]
            print(f"    🔧 调整后形状: {predicted_action.shape}")
        
        # 检查并对齐数据格式（处理normalization一致性）
        aligned_predicted, aligned_gt, alignment_info = self.check_and_align_data_format(
            predicted_action, gt_actions
        )
        
        # 计算逐步差异
        step_wise_diff = np.abs(aligned_predicted - aligned_gt)  # (16, 18)
        step_wise_pose_diff = step_wise_diff[:, :6]  # (16, 6) 姿态部分
        step_wise_hand_diff = step_wise_diff[:, 6:]  # (16, 12) 手部部分
        
        # 计算各种统计指标
        # 分时间段统计
        execution_steps = 8  # action_horizon
        execution_diff = step_wise_diff[:execution_steps]  # 前8步执行段
        prediction_diff = step_wise_diff[execution_steps:]  # 后8步预测段
        
        results = {
            'episode_name': episode_name,
            'step_idx': step_idx,
            'raw_data': {
                'image': raw_image,
                'pose': raw_pose,
                'proprioception': raw_data['proprioception'][step_idx] if not self.policy.model_cfg.dataset.skip_proprioception else None,
                'fsr': raw_data['fsr'][step_idx] if self.policy.model_cfg.dataset.enable_fsr else None,
            },
            'model_config': {
                'skip_proprioception': self.policy.model_cfg.dataset.skip_proprioception,
                'enable_fsr': self.policy.model_cfg.dataset.enable_fsr,
                'camera_ids': getattr(self.policy.model_cfg.dataset, 'load_camera_ids', [0]),
                'bgr2rgb': self.policy.model_cfg.dataset.bgr2rgb,
            },
            'ground_truth_actions': gt_actions,  # 完整16步
            'predicted_actions': predicted_action,  # 完整16步
            'step_wise_diff': step_wise_diff,
            
            # 整体统计
            'overall_rmse': np.sqrt(np.mean(step_wise_diff**2)),
            'overall_mae': np.mean(step_wise_diff),
            'overall_max_error': np.max(step_wise_diff),
            
            # 分段统计 - 执行段 (1-8步)
            'execution_rmse': np.sqrt(np.mean(execution_diff**2)),
            'execution_mae': np.mean(execution_diff),
            'execution_max_error': np.max(execution_diff),
            
            # 分段统计 - 预测段 (9-16步)  
            'prediction_rmse': np.sqrt(np.mean(prediction_diff**2)),
            'prediction_mae': np.mean(prediction_diff),
            'prediction_max_error': np.max(prediction_diff),
            
            # 分模态统计
            'pose_rmse': np.sqrt(np.mean(step_wise_pose_diff**2)),
            'hand_rmse': np.sqrt(np.mean(step_wise_hand_diff**2)),
            
            # 时序分析
            'step_rmse': [np.sqrt(np.mean(step_wise_diff[i]**2)) for i in range(len(step_wise_diff))],
            'step_mae': [np.mean(step_wise_diff[i]) for i in range(len(step_wise_diff))],
        }
        
        print(f"📈 时序预测质量分析:")
        print(f"  🎯 整体性能:")
        print(f"    RMSE: {results['overall_rmse']:.4f}")
        print(f"    MAE:  {results['overall_mae']:.4f}")
        print(f"    Max:  {results['overall_max_error']:.4f}")
        print(f"  🏃 执行段性能 (步骤1-8):")
        print(f"    RMSE: {results['execution_rmse']:.4f}")
        print(f"    MAE:  {results['execution_mae']:.4f}")
        print(f"  🔮 预测段性能 (步骤9-16):")
        print(f"    RMSE: {results['prediction_rmse']:.4f}")
        print(f"    MAE:  {results['prediction_mae']:.4f}")
        print(f"  🤖 分模态性能:")
        print(f"    Pose RMSE: {results['pose_rmse']:.4f}")
        print(f"    Hand RMSE: {results['hand_rmse']:.4f}")
        
        # 添加初始绝对pose到结果中
        results['initial_absolute_pose'] = initial_absolute_pose
        
        return results
    
    def find_matching_sample_index(self, episode_idx, step_idx):
        """
        找到与给定episode和step对应的dataset样本索引
        
        Args:
            episode_idx: episode索引
            step_idx: 时间步索引
            
        Returns:
            int: dataset中对应的样本索引
        """
        print(f"    🔍 查找匹配样本: episode_{episode_idx}, step_{step_idx}")
        
        # 方法1: 使用dataset的indices信息反推
        if hasattr(self.dataset, 'indices') and hasattr(self.dataset.buffer, 'eps_end'):
            try:
                eps_ends = self.dataset.buffer.eps_end
                
                # 计算目标在buffer中的全局位置
                if episode_idx == 0:
                    episode_start_in_buffer = 0
                else:
                    episode_start_in_buffer = eps_ends[episode_idx - 1]
                
                target_global_idx = episode_start_in_buffer + step_idx
                
                # 在dataset的indices中找到对应的样本
                for sample_idx, (buffer_start, buffer_end, sample_start, sample_end) in enumerate(self.dataset.indices):
                    # 检查target_global_idx是否在这个样本的范围内
                    if buffer_start <= target_global_idx < buffer_end:
                        print(f"      ✅ 方法1成功: 样本{sample_idx}, buffer范围[{buffer_start}:{buffer_end}]")
                        return sample_idx
                        
            except Exception as e:
                print(f"      ❌ 方法1失败: {e}")
        
        # 方法2: 基于episode统计信息估算
        try:
            # 获取当前episode的长度信息
            root = zarr.open(self.dataset_path, mode="r")
            episodes = sorted(list(root.group_keys()))
            
            if episode_idx < len(episodes):
                episode_name = episodes[episode_idx]
                episode_length = len(root[f"{episode_name}/pose"])
                
                # 计算前面所有episode贡献的样本数
                total_samples_before = 0
                for prev_ep_idx in range(episode_idx):
                    prev_episode_name = episodes[prev_ep_idx]
                    prev_length = len(root[f"{prev_episode_name}/pose"])
                    # 每个episode的样本数 = max(0, length - pred_horizon + 1)
                    samples_from_prev_ep = max(0, prev_length - 16 + 1)
                    total_samples_before += samples_from_prev_ep
                
                # 当前episode内的样本偏移
                sample_offset_in_episode = min(step_idx, max(0, episode_length - 16))
                
                estimated_sample_idx = total_samples_before + sample_offset_in_episode
                estimated_sample_idx = min(estimated_sample_idx, len(self.dataset) - 1)
                
                print(f"      ✅ 方法2估算: episode长度{episode_length}, 估算索引{estimated_sample_idx}")
                return estimated_sample_idx
                
        except Exception as e:
            print(f"      ❌ 方法2失败: {e}")
        
        # 方法3: 简单估算（回退方案）
        estimated_idx = min(episode_idx * 15 + step_idx, len(self.dataset) - 1)
        estimated_idx = max(0, estimated_idx)
        print(f"      🔄 方法3回退: 简单估算索引{estimated_idx}")
        
        return estimated_idx
    
    def check_and_align_data_format(self, predicted_actions, gt_actions):
        """
        检查并对齐预测数据和GT数据的格式
        处理normalization一致性问题
        
        Args:
            predicted_actions: 模型预测的actions (16, 18)
            gt_actions: ground truth actions (16, 18)
            
        Returns:
            tuple: (aligned_predicted, aligned_gt, alignment_info)
        """
        print(f"\n🔍 数据格式对齐检查:")
        alignment_info = {}
        
        # 1. 检查数据范围，判断normalization状态
        pred_range = (predicted_actions.min(), predicted_actions.max())
        gt_range = (gt_actions.min(), gt_actions.max())
        
        print(f"  📊 预测数据范围: [{pred_range[0]:.3f}, {pred_range[1]:.3f}]")
        print(f"  📊 GT数据范围:   [{gt_range[0]:.3f}, {gt_range[1]:.3f}]")
        
        # 判断数据是否需要denormalization
        # normalized数据通常在[-3, 3]范围内，原始数据范围更大
        pred_is_normalized = abs(pred_range[0]) < 5 and abs(pred_range[1]) < 5
        gt_is_normalized = abs(gt_range[0]) < 5 and abs(gt_range[1]) < 5
        
        alignment_info['pred_normalized'] = pred_is_normalized
        alignment_info['gt_normalized'] = gt_is_normalized
        
        print(f"  🔬 预测数据状态: {'Normalized' if pred_is_normalized else 'Raw'}")
        print(f"  🔬 GT数据状态:   {'Normalized' if gt_is_normalized else 'Raw'}")
        
        # 2. 对齐数据格式
        aligned_predicted = predicted_actions.copy()
        aligned_gt = gt_actions.copy()
        
        if pred_is_normalized and gt_is_normalized:
            # 两者都是normalized，直接比较
            print(f"  ✅ 两者都是normalized数据，直接比较")
            alignment_info['comparison_space'] = 'normalized'
            
        elif not pred_is_normalized and not gt_is_normalized:
            # 两者都是原始数据，直接比较
            print(f"  ✅ 两者都是原始数据，直接比较")  
            alignment_info['comparison_space'] = 'raw'
            
        elif pred_is_normalized and not gt_is_normalized:
            # 预测是normalized，GT是原始数据 - 需要denormalize预测
            print(f"  🔄 预测数据normalized，GT是原始数据，尝试denormalize预测")
            try:
                # 尝试使用数据集的统计信息denormalize
                if hasattr(self.dataset, 'stats'):
                    # 处理pose部分 (0:6)
                    if 'relative_pose' in self.dataset.stats:
                        pose_stats = self.dataset.stats['relative_pose']
                        aligned_predicted[:, :6] = self.denormalize_data(
                            aligned_predicted[:, :6], pose_stats
                        )
                    
                    # 处理hand部分 (6:18) 
                    if 'hand_action' in self.dataset.stats:
                        hand_stats = self.dataset.stats['hand_action']
                        aligned_predicted[:, 6:] = self.denormalize_data(
                            aligned_predicted[:, 6:], hand_stats
                        )
                    
                    print(f"    ✅ 预测数据已denormalize")
                    alignment_info['comparison_space'] = 'raw'
                    
                else:
                    print(f"    ⚠️ 无法获取统计信息，保持normalized空间比较")
                    alignment_info['comparison_space'] = 'normalized'
                    
            except Exception as e:
                print(f"    ❌ Denormalize失败: {e}，保持normalized空间比较")
                alignment_info['comparison_space'] = 'normalized'
                
        else:
            # GT是normalized，预测是原始数据 - 这种情况很少见
            print(f"  ⚠️ 不常见的情况：GT normalized，预测原始数据")
            alignment_info['comparison_space'] = 'mixed'
        
        # 3. 验证对齐后的数据
        final_pred_range = (aligned_predicted.min(), aligned_predicted.max())
        final_gt_range = (aligned_gt.min(), aligned_gt.max())
        
        print(f"  📈 对齐后预测范围: [{final_pred_range[0]:.3f}, {final_pred_range[1]:.3f}]")
        print(f"  📈 对齐后GT范围:   [{final_gt_range[0]:.3f}, {final_gt_range[1]:.3f}]")
        
        return aligned_predicted, aligned_gt, alignment_info
    
    def denormalize_data(self, normalized_data, stats):
        """反normalize数据到原始范围"""
        data_min = stats['min']
        data_max = stats['max']
        
        # 从[-1, 1]恢复到[min, max]
        denormalized = (normalized_data + 1) / 2.0 * (data_max - data_min) + data_min
        return denormalized
    
    def reconstruct_absolute_poses(self, relative_poses, initial_pose):
        """
        从相对变换重建绝对pose序列
        
        Args:
            relative_poses: 相对pose序列 (16, 6) - 相对于第一帧的变换
            initial_pose: 初始绝对pose (6,) - 第一帧的绝对坐标
            
        Returns:
            absolute_poses: 绝对pose序列 (16, 6)
        """
        print(f"🔄 重建绝对pose轨迹...")
        
        # 初始pose的变换矩阵
        T0 = vec6dof_to_homogeneous_matrix(
            translation=initial_pose[:3], 
            rotation_vector=initial_pose[3:]
        )
        
        absolute_poses = []
        for i in range(len(relative_poses)):
            # 相对变换矩阵
            T_rel = vec6dof_to_homogeneous_matrix(
                translation=relative_poses[i, :3],
                rotation_vector=relative_poses[i, 3:]
            )
            
            # 绝对变换 = 初始变换 * 相对变换
            T_abs = T0 @ T_rel
            
            # 转回6DOF
            abs_pose = homogeneous_matrix_to_6dof(T_abs)
            absolute_poses.append(abs_pose)
        
        absolute_poses = np.array(absolute_poses, dtype=np.float32)
        print(f"  ✅ 重建完成: {absolute_poses.shape}")
        return absolute_poses
    
    def denormalize_relative_poses(self, normalized_relative_poses):
        """
        反归一化相对pose数据
        
        Args:
            normalized_relative_poses: 归一化的相对pose (16, 6)
            
        Returns:
            denormalized_poses: 反归一化的相对pose (16, 6)
        """
        if hasattr(self.dataset, 'stats') and 'relative_pose' in self.dataset.stats:
            stats = self.dataset.stats['relative_pose']
            # 从[-1, 1]恢复到原始范围
            data_min = stats['min']
            data_max = stats['max']
            denormalized = (normalized_relative_poses + 1) / 2.0 * (data_max - data_min) + data_min
            return denormalized
        else:
            print("⚠️ 未找到relative_pose统计信息，返回原始数据")
            return normalized_relative_poses
        
    def visualize_dual_coordinate_analysis(self, results, save_path=None):
        """
        双重坐标系可视化：绝对轨迹 + 相对误差分析
        
        Args:
            results: single_step_prediction的返回结果  
            save_path: 保存路径（可选）
        """
        print("\n📈 Generating dual coordinate system analysis...")
        
        gt_actions = results['ground_truth_actions']  # (16, 18)
        pred_actions = results['predicted_actions']   # (16, 18)
        
        # 分离pose和hand_action
        gt_relative_poses = gt_actions[:, :6]      # (16, 6) - GT相对pose
        pred_relative_poses = pred_actions[:, :6]  # (16, 6) - 预测相对pose  
        gt_hand_actions = gt_actions[:, 6:]        # (16, 12) - GT手部动作
        pred_hand_actions = pred_actions[:, 6:]    # (16, 12) - 预测手部动作
        
        # 反归一化相对pose
        gt_relative_denorm = self.denormalize_relative_poses(gt_relative_poses)
        pred_relative_denorm = self.denormalize_relative_poses(pred_relative_poses)
        
        # 获取初始绝对pose用于重建
        initial_pose = results.get('initial_absolute_pose')
        if initial_pose is None:
            print("⚠️ 未找到初始绝对pose，使用零初始化")
            initial_pose = np.zeros(6)
        
        # 重建绝对轨迹
        gt_absolute_poses = self.reconstruct_absolute_poses(gt_relative_denorm, initial_pose)
        pred_absolute_poses = self.reconstruct_absolute_poses(pred_relative_denorm, initial_pose)
        
        # 创建4x3的子图布局
        fig, axes = plt.subplots(4, 3, figsize=(24, 20))
        
        # 时间步轴
        time_steps = np.arange(1, 17)
        execution_steps = time_steps[:8]  # 执行段 (1-8步)
        prediction_steps = time_steps[8:]  # 预测段 (9-16步)
        
        # ============ 第一行：绝对轨迹对比 ============
        
        # 1.1 绝对位置轨迹 (XYZ)
        colors = ['red', 'green', 'blue']
        position_labels = ['X', 'Y', 'Z']
        for i in range(3):
            color = colors[i]
            axes[0, 0].plot(time_steps, gt_absolute_poses[:, i], 'o-', color=color, alpha=0.8, 
                           linewidth=2, label=f'GT {position_labels[i]}')
            axes[0, 0].plot(time_steps, pred_absolute_poses[:, i], 's--', color=color, alpha=0.8, 
                           linewidth=2, label=f'Pred {position_labels[i]}')
        
        axes[0, 0].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Absolute Position Trajectories (XYZ)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 1.2 绝对旋转轨迹 (Rotation Vectors)
        rotation_labels = ['Rx', 'Ry', 'Rz']
        for i in range(3):
            color = colors[i]
            axes[0, 1].plot(time_steps, gt_absolute_poses[:, 3+i], 'o-', color=color, alpha=0.8,
                           linewidth=2, label=f'GT {rotation_labels[i]}')
            axes[0, 1].plot(time_steps, pred_absolute_poses[:, 3+i], 's--', color=color, alpha=0.8,
                           linewidth=2, label=f'Pred {rotation_labels[i]}')
        
        axes[0, 1].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('Absolute Rotation Trajectories (Rot Vectors)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Rotation (rad)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 1.3 3D轨迹图 
        ax_3d = fig.add_subplot(4, 3, 3, projection='3d')
        ax_3d.plot(gt_absolute_poses[:, 0], gt_absolute_poses[:, 1], gt_absolute_poses[:, 2], 
                  'o-', color='blue', alpha=0.8, linewidth=2, markersize=4, label='GT Trajectory')
        ax_3d.plot(pred_absolute_poses[:, 0], pred_absolute_poses[:, 1], pred_absolute_poses[:, 2], 
                  's--', color='red', alpha=0.8, linewidth=2, markersize=4, label='Pred Trajectory')
        
        # 标记起始点和结束点
        ax_3d.scatter(*gt_absolute_poses[0, :3], color='green', s=100, marker='o', label='Start')
        ax_3d.scatter(*gt_absolute_poses[-1, :3], color='orange', s=100, marker='x', label='End')
        
        ax_3d.set_title('3D Absolute Trajectory', fontsize=14, fontweight='bold')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')  
        ax_3d.set_zlabel('Z (m)')
        ax_3d.legend()
        ax_3d.grid(True, alpha=0.3)
        
        # ============ 第二行：相对误差分析 ============
        
        # 2.1 相对位置误差
        pos_errors = np.abs(gt_relative_denorm[:, :3] - pred_relative_denorm[:, :3])
        for i in range(3):
            color = colors[i]
            axes[1, 0].plot(time_steps, pos_errors[:, i], 'o-', color=color, alpha=0.8,
                           linewidth=2, label=f'Pos {position_labels[i]} Error')
        
        axes[1, 0].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[1, 0].fill_between(execution_steps, 0, np.max(pos_errors[:8], axis=1), 
                               alpha=0.2, color='blue', label='Execution Segment')
        axes[1, 0].fill_between(prediction_steps, 0, np.max(pos_errors[8:], axis=1), 
                               alpha=0.2, color='orange', label='Prediction Segment')
        axes[1, 0].set_title('Relative Position Errors', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Position Error (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2.2 相对旋转误差
        rot_errors = np.abs(gt_relative_denorm[:, 3:] - pred_relative_denorm[:, 3:])
        for i in range(3):
            color = colors[i]
            axes[1, 1].plot(time_steps, rot_errors[:, i], 'o-', color=color, alpha=0.8,
                           linewidth=2, label=f'Rot {rotation_labels[i]} Error')
        
        axes[1, 1].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[1, 1].fill_between(execution_steps, 0, np.max(rot_errors[:8], axis=1), 
                               alpha=0.2, color='blue', label='Execution Segment')
        axes[1, 1].fill_between(prediction_steps, 0, np.max(rot_errors[8:], axis=1), 
                               alpha=0.2, color='orange', label='Prediction Segment')
        axes[1, 1].set_title('Relative Rotation Errors', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Rotation Error (rad)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 2.3 手部动作误差
        hand_errors = np.abs(gt_hand_actions - pred_hand_actions)
        hand_error_mean = np.mean(hand_errors, axis=1)  # 平均手部误差
        axes[1, 2].plot(time_steps, hand_error_mean, 'o-', color='purple', alpha=0.8,
                       linewidth=2, markersize=6, label='Mean Hand Error')
        
        axes[1, 2].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[1, 2].fill_between(execution_steps, 0, hand_error_mean[:8], 
                               alpha=0.3, color='blue', label='Execution Segment')
        axes[1, 2].fill_between(prediction_steps, 0, hand_error_mean[8:], 
                               alpha=0.3, color='orange', label='Prediction Segment')
        axes[1, 2].set_title('Hand Action Errors', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Mean Absolute Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # ============ 第三行：执行段vs预测段对比 ============
        
        # 3.1 绝对轨迹误差对比
        abs_pos_errors = np.abs(gt_absolute_poses - pred_absolute_poses)
        exec_pos_err = np.mean(abs_pos_errors[:8, :3], axis=0)  # 执行段位置误差
        pred_pos_err = np.mean(abs_pos_errors[8:, :3], axis=0)  # 预测段位置误差
        exec_rot_err = np.mean(abs_pos_errors[:8, 3:], axis=0)  # 执行段旋转误差
        pred_rot_err = np.mean(abs_pos_errors[8:, 3:], axis=0)  # 预测段旋转误差
        
        pos_dims = np.arange(3)
        width = 0.35
        axes[2, 0].bar(pos_dims - width/2, exec_pos_err, width, 
                      label='Execution Segment (Steps 1-8)', alpha=0.7, color='blue')
        axes[2, 0].bar(pos_dims + width/2, pred_pos_err, width, 
                      label='Prediction Segment (Steps 9-16)', alpha=0.7, color='orange')
        axes[2, 0].set_title('Absolute Position Error Comparison', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Position Dimension')
        axes[2, 0].set_ylabel('Mean Absolute Error (m)')
        axes[2, 0].set_xticks(pos_dims)
        axes[2, 0].set_xticklabels(['X', 'Y', 'Z'])
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 3.2 绝对旋转误差对比
        rot_dims = np.arange(3)
        axes[2, 1].bar(rot_dims - width/2, exec_rot_err, width, 
                      label='Execution Segment (Steps 1-8)', alpha=0.7, color='blue')
        axes[2, 1].bar(rot_dims + width/2, pred_rot_err, width, 
                      label='Prediction Segment (Steps 9-16)', alpha=0.7, color='orange')
        axes[2, 1].set_title('Absolute Rotation Error Comparison', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Rotation Dimension')
        axes[2, 1].set_ylabel('Mean Absolute Error (rad)')
        axes[2, 1].set_xticks(rot_dims)
        axes[2, 1].set_xticklabels(['Rx', 'Ry', 'Rz'])
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 3.3 累积误差趋势
        pose_step_errors = np.mean(abs_pos_errors, axis=1)  # 每步的平均pose误差
        hand_step_errors = np.mean(hand_errors, axis=1)     # 每步的平均手部误差
        cumulative_pose_errors = np.cumsum(pose_step_errors)
        cumulative_hand_errors = np.cumsum(hand_step_errors)
        
        axes[2, 2].plot(time_steps, cumulative_pose_errors, 'o-', color='blue', 
                       linewidth=2, markersize=4, label='Cumulative Pose Error')
        axes[2, 2].plot(time_steps, cumulative_hand_errors, 's-', color='red', 
                       linewidth=2, markersize=4, label='Cumulative Hand Error')
        axes[2, 2].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[2, 2].set_title('Cumulative Error Trends', fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('Time Step')
        axes[2, 2].set_ylabel('Cumulative Error')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # ============ 第四行：统计摘要与热图 ============
        
        # 4.1 统计分析摘要
        exec_stats = results.get('execution_segment_stats', {})
        pred_stats = results.get('prediction_segment_stats', {})
        overall_stats = results.get('overall_stats', {})
        
        stats_text = f"""
Dual Coordinate Analysis Summary:

ABSOLUTE TRAJECTORY METRICS:
  Position RMSE: {np.sqrt(np.mean(abs_pos_errors**2)):.4f} m
  Rotation RMSE: {np.sqrt(np.mean(abs_pos_errors[:, 3:]**2)):.4f} rad
  
RELATIVE ERROR ANALYSIS:
  Execution Segment (Steps 1-8):
    Pos RMSE: {np.sqrt(np.mean(pos_errors[:8]**2)):.4f} m
    Rot RMSE: {np.sqrt(np.mean(rot_errors[:8]**2)):.4f} rad
    
  Prediction Segment (Steps 9-16):
    Pos RMSE: {np.sqrt(np.mean(pos_errors[8:]**2)):.4f} m
    Rot RMSE: {np.sqrt(np.mean(rot_errors[8:]**2)):.4f} rad

HAND ACTION PERFORMANCE:
  Overall RMSE: {np.sqrt(np.mean(hand_errors**2)):.4f}
  Execution RMSE: {np.sqrt(np.mean(hand_errors[:8]**2)):.4f}
  Prediction RMSE: {np.sqrt(np.mean(hand_errors[8:]**2)):.4f}
        """
        
        axes[3, 0].text(0.05, 0.95, stats_text, transform=axes[3, 0].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[3, 0].set_title('Statistical Analysis Summary', fontsize=14, fontweight='bold')
        axes[3, 0].axis('off')
        
        # 4.2 误差分布热图
        all_errors = np.concatenate([
            abs_pos_errors.T,           # 绝对pose误差 (6, 16)
            hand_errors.T              # 手部误差 (12, 16)
        ], axis=0)  # (18, 16)
        
        im = axes[3, 1].imshow(all_errors, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[3, 1].axhline(y=5.5, color='white', linestyle='-', linewidth=2)  # pose和hand分界线
        axes[3, 1].axvline(x=7.5, color='red', linestyle=':', alpha=0.7, linewidth=2)  # 执行|预测边界
        
        # 添加标签
        axes[3, 1].set_xticks(range(0, 16, 2))
        axes[3, 1].set_xticklabels([f'{i+1}' for i in range(0, 16, 2)])
        axes[3, 1].set_ylabel('Action Dimension')
        axes[3, 1].set_xlabel('Time Step')
        axes[3, 1].set_title('Error Distribution Heatmap\n(Top: Pose, Bottom: Hand)', fontsize=14, fontweight='bold')
        
        # 添加colorbar
        plt.colorbar(im, ax=axes[3, 1])
        
        # 4.3 轨迹距离分析
        # 计算GT和预测轨迹之间的欧几里得距离
        traj_distances = np.sqrt(np.sum((gt_absolute_poses[:, :3] - pred_absolute_poses[:, :3])**2, axis=1))
        
        axes[3, 2].plot(time_steps, traj_distances, 'o-', color='purple', 
                       linewidth=2, markersize=6, label='3D Trajectory Distance')
        axes[3, 2].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[3, 2].fill_between(execution_steps, 0, traj_distances[:8], 
                               alpha=0.3, color='blue', label='Execution Segment')
        axes[3, 2].fill_between(prediction_steps, 0, traj_distances[8:], 
                               alpha=0.3, color='orange', label='Prediction Segment')
        axes[3, 2].set_title('3D Trajectory Distance', fontsize=14, fontweight='bold')
        axes[3, 2].set_xlabel('Time Step')
        axes[3, 2].set_ylabel('Distance (m)')
        axes[3, 2].legend()
        axes[3, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            dual_save_path = save_path.replace('.png', '_dual_coordinate.png')
            plt.savefig(dual_save_path, dpi=150, bbox_inches='tight')
            print(f"📁 Dual coordinate analysis saved to: {dual_save_path}")
        
        plt.show()

    def visualize_trajectory_analysis(self, results, save_path=None):
        """
        保持原有的轨迹分析可视化（为了向后兼容）
        """
        print("\n📈 Generating 16-step trajectory analysis visualization...")
        
        gt_actions = results['ground_truth_actions']  # (16, 18)
        pred_actions = results['predicted_actions']   # (16, 18)
        
        # 创建3x3的子图布局
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 时间步轴
        time_steps = np.arange(1, 17)
        execution_steps = time_steps[:8]  # 执行段 (1-8步)
        prediction_steps = time_steps[8:]  # 预测段 (9-16步)
        
        # 1. 整体轨迹对比 - Pose (前6维)
        for i in range(6):
            color = plt.cm.tab10(i)
            axes[0, 0].plot(time_steps, gt_actions[:, i], 'o-', color=color, alpha=0.7, label=f'GT Pose {i}')
            axes[0, 0].plot(time_steps, pred_actions[:, i], 's--', color=color, alpha=0.7, label=f'Pred Pose {i}')
        
        axes[0, 0].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, label='Execution|Prediction Boundary')
        axes[0, 0].set_title('16-Step Pose Trajectory Comparison (6D)')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Pose Value')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 手部动作轨迹 - 分批显示
        # 前6个手部关节
        for i in range(6):
            color = plt.cm.tab10(i)
            axes[0, 1].plot(time_steps, gt_actions[:, 6+i], 'o-', color=color, alpha=0.7, label=f'GT Joint {i}')
            axes[0, 1].plot(time_steps, pred_actions[:, 6+i], 's--', color=color, alpha=0.7, label=f'Pred Joint {i}')
        
        axes[0, 1].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, label='Execution|Prediction Boundary')
        axes[0, 1].set_title('16-Step Hand Trajectory Comparison (First 6 Joints)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Joint Value')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 后6个手部关节
        for i in range(6):
            color = plt.cm.tab10(i)
            axes[0, 2].plot(time_steps, gt_actions[:, 12+i], 'o-', color=color, alpha=0.7, label=f'GT Joint {6+i}')
            axes[0, 2].plot(time_steps, pred_actions[:, 12+i], 's--', color=color, alpha=0.7, label=f'Pred Joint {6+i}')
        
        axes[0, 2].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, label='Execution|Prediction Boundary')
        axes[0, 2].set_title('16-Step Hand Trajectory Comparison (Last 6 Joints)')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('Joint Value')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 执行段vs预测段误差对比
        execution_errors = np.abs(gt_actions[:8] - pred_actions[:8])  # (8, 18)
        prediction_errors = np.abs(gt_actions[8:] - pred_actions[8:])  # (8, 18)
        
        # 分模态显示误差
        pose_exec_err = np.mean(execution_errors[:, :6], axis=0)  # (6,)
        pose_pred_err = np.mean(prediction_errors[:, :6], axis=0)  # (6,)
        hand_exec_err = np.mean(execution_errors[:, 6:], axis=0)   # (12,)
        hand_pred_err = np.mean(prediction_errors[:, 6:], axis=0)  # (12,)
        
        pose_dims = np.arange(6)
        width = 0.35
        axes[1, 0].bar(pose_dims - width/2, pose_exec_err, width, label='Execution Segment (Steps 1-8)', alpha=0.7, color='blue')
        axes[1, 0].bar(pose_dims + width/2, pose_pred_err, width, label='Prediction Segment (Steps 9-16)', alpha=0.7, color='orange')
        axes[1, 0].set_title('Pose Error Comparison: Execution vs Prediction')
        axes[1, 0].set_xlabel('Pose Dimension')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 手部关节误差对比
        hand_dims = np.arange(12)
        axes[1, 1].bar(hand_dims - width/2, hand_exec_err, width, label='Execution Segment (Steps 1-8)', alpha=0.7, color='blue')
        axes[1, 1].bar(hand_dims + width/2, hand_pred_err, width, label='Prediction Segment (Steps 9-16)', alpha=0.7, color='orange')
        axes[1, 1].set_title('Hand Joint Error Comparison: Execution vs Prediction')
        axes[1, 1].set_xlabel('Joint Index')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 时序误差趋势
        step_wise_errors = np.mean(np.abs(gt_actions - pred_actions), axis=1)  # (16,) - 每步的平均误差
        axes[1, 2].plot(time_steps, step_wise_errors, 'o-', color='purple', linewidth=2, markersize=6)
        axes[1, 2].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, label='Execution|Prediction Boundary')
        axes[1, 2].fill_between(execution_steps, 0, step_wise_errors[:8], alpha=0.3, color='blue', label='Execution Segment')
        axes[1, 2].fill_between(prediction_steps, 0, step_wise_errors[8:], alpha=0.3, color='orange', label='Prediction Segment')
        axes[1, 2].set_title('Temporal Error Trend')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Mean Absolute Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 累积误差分析
        cumulative_errors = np.cumsum(step_wise_errors)
        axes[2, 0].plot(time_steps, cumulative_errors, 'o-', color='red', linewidth=2, markersize=6)
        axes[2, 0].axvline(x=8.5, color='red', linestyle=':', alpha=0.7, label='Execution|Prediction Boundary')
        axes[2, 0].set_title('Cumulative Error Trend')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Cumulative Absolute Error')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 统计分析摘要
        exec_stats = results.get('execution_segment_stats', {})
        pred_stats = results.get('prediction_segment_stats', {})
        overall_stats = results.get('overall_stats', {})
        
        stats_text = f"""
16-Step Trajectory Analysis:

Execution Segment (Steps 1-8):
  RMSE: {exec_stats.get('rmse', 0):.4f}
  MAE:  {exec_stats.get('mae', 0):.4f}
  Max:  {exec_stats.get('max_err', 0):.4f}

Prediction Segment (Steps 9-16):
  RMSE: {pred_stats.get('rmse', 0):.4f}
  MAE:  {pred_stats.get('mae', 0):.4f}
  Max:  {pred_stats.get('max_err', 0):.4f}

Overall Performance:
  RMSE: {overall_stats.get('rmse', 0):.4f}
  MAE:  {overall_stats.get('mae', 0):.4f}
  Max:  {overall_stats.get('max_err', 0):.4f}

Multi-Modal Performance:
  Pose RMSE: {results.get('pose_rmse', 0):.4f}
  Hand RMSE: {results.get('hand_rmse', 0):.4f}
        """
        
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axes[2, 1].set_title('Statistical Analysis Summary')
        axes[2, 1].axis('off')
        
        # 9. 动作分布热图
        # 将16步数据reshape为热图
        combined_actions = np.concatenate([gt_actions.T, pred_actions.T], axis=0)  # (36, 16)
        
        im = axes[2, 2].imshow(combined_actions, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[2, 2].axhline(y=17.5, color='white', linestyle='-', linewidth=2)  # GT和Pred分界线
        axes[2, 2].axvline(x=7.5, color='red', linestyle=':', alpha=0.7)  # 执行|预测边界
        
        # 添加标签
        axes[2, 2].set_xticks(range(0, 16, 2))
        axes[2, 2].set_xticklabels([f'{i+1}' for i in range(0, 16, 2)])
        axes[2, 2].set_ylabel('Action Dimension')
        axes[2, 2].set_xlabel('Time Step')
        axes[2, 2].set_title('16-Step Action Heatmap\n(Top: GT, Bottom: Pred)')
        
        # 添加colorbar
        plt.colorbar(im, ax=axes[2, 2])
        
        plt.tight_layout()
        
        if save_path:
            trajectory_save_path = save_path.replace('.png', '_trajectory.png')
            plt.savefig(trajectory_save_path, dpi=150, bbox_inches='tight')
            print(f"📁 16-step trajectory analysis saved to: {trajectory_save_path}")
        
        plt.show()

    def visualize_results(self, results, save_path=None):
        """
        可视化调试结果，重点显示动作预测误差分析
        
        Args:
            results: single_step_prediction的返回结果
            save_path: 保存路径（可选）
        """
        print("\n📈 Generating action prediction analysis visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 动作对比 - 使用16步序列
        gt_actions = results['ground_truth_actions']  # (16, 18)
        pred_actions = results['predicted_actions']   # (16, 18)
        
        # 为了兼容原有可视化，取第一步
        gt_action = gt_actions[0]   # (18,)
        pred_action = pred_actions[0]  # (18,)
        
        # 时间序列动作对比 - 显示整个16步序列的前6个维度（姿态）
        steps = np.arange(16)
        for i in range(6):
            alpha = 0.7 if i < 3 else 0.4  # 前3个维度更突出
            axes[0, 0].plot(steps, gt_actions[:, i], '--', alpha=alpha, label=f'GT Pose{i}' if i < 3 else None)
            axes[0, 0].plot(steps, pred_actions[:, i], '-', alpha=alpha, label=f'Pred Pose{i}' if i < 3 else None)
        axes[0, 0].set_title('Pose Actions Over 16 Steps')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Action Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 手部动作时间序列（显示前6个手部关节）
        for i in range(6):
            alpha = 0.7 if i < 3 else 0.4
            axes[0, 1].plot(steps, gt_actions[:, 6+i], '--', alpha=alpha, label=f'GT Hand{i}' if i < 3 else None)
            axes[0, 1].plot(steps, pred_actions[:, 6+i], '-', alpha=alpha, label=f'Pred Hand{i}' if i < 3 else None)
        axes[0, 1].set_title('Hand Actions Over 16 Steps (First 6 Joints)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 每步误差分布
        step_errors = np.linalg.norm(gt_actions - pred_actions, axis=1)
        axes[0, 2].plot(steps, step_errors, 'o-', color='red', linewidth=2)
        axes[0, 2].set_title('L2 Error Per Time Step')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('L2 Error')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 姿态动作对比（第一步）
        pose_indices = np.arange(6)
        axes[1, 0].bar(pose_indices - 0.2, gt_action[:6], 0.4, label='Ground Truth', alpha=0.7)
        axes[1, 0].bar(pose_indices + 0.2, pred_action[:6], 0.4, label='Predicted', alpha=0.7)
        axes[1, 0].set_title('Pose Action Comparison (Step 1)')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 手部动作对比（第一步）
        hand_indices = np.arange(12)
        axes[1, 1].bar(hand_indices - 0.2, gt_action[6:], 0.4, label='Ground Truth', alpha=0.7)
        axes[1, 1].bar(hand_indices + 0.2, pred_action[6:], 0.4, label='Predicted', alpha=0.7)
        axes[1, 1].set_title('Hand Action Comparison (Step 1)')
        axes[1, 1].set_xlabel('Joint')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 差异分析热图 - 16步 x 18维
        action_diff_matrix = np.abs(gt_actions - pred_actions)  # (16, 18)
        im = axes[1, 2].imshow(action_diff_matrix.T, cmap='hot', aspect='auto')
        axes[1, 2].set_title('Action Difference Heatmap')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Action Dimension')
        axes[1, 2].axhline(y=5.5, color='white', linestyle='--', alpha=0.7)
        plt.colorbar(im, ax=axes[1, 2])
        
        # 计算统计信息
        pose_diff_norm = np.linalg.norm(gt_actions[:, :6] - pred_actions[:, :6])
        hand_diff_norm = np.linalg.norm(gt_actions[:, 6:] - pred_actions[:, 6:])
        max_action_diff = np.max(action_diff_matrix)
        mean_action_diff = np.mean(action_diff_matrix)
        
        print(f"\n📊 Action Prediction Statistics:")
        print(f"  Pose L2 Error (16 steps): {pose_diff_norm:.3f}")
        print(f"  Hand L2 Error (16 steps): {hand_diff_norm:.3f}")
        print(f"  Max Single Diff: {max_action_diff:.3f}")
        print(f"  Mean Diff: {mean_action_diff:.3f}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📁 结果已保存到: {save_path}")
        
        plt.show()
        
    def batch_validation(self, num_samples=10, episode_range=None):
        """
        批量验证多个样本
        
        Args:
            num_samples: 验证样本数量
            episode_range: episode范围 (start, end)
        """
        print(f"\n🔄 开始批量验证 ({num_samples} 个样本)")
        
        results_summary = {
            'overall_rmse': [],
            'overall_mae': [],
            'execution_rmse': [],
            'prediction_rmse': [],
            'pose_rmse': [],
            'hand_rmse': [],
            'overall_max_error': [],
        }
        
        episode_start = episode_range[0] if episode_range else 0
        episode_end = episode_range[1] if episode_range else min(5, num_samples)
        
        for i in range(num_samples):
            episode_idx = episode_start + (i % (episode_end - episode_start))
            step_idx = np.random.randint(0, 50)  # 随机选择时间步
            
            try:
                results = self.single_step_prediction(episode_idx, step_idx)
                
                results_summary['overall_rmse'].append(results['overall_rmse'])
                results_summary['overall_mae'].append(results['overall_mae'])
                results_summary['execution_rmse'].append(results['execution_rmse'])
                results_summary['prediction_rmse'].append(results['prediction_rmse'])
                results_summary['pose_rmse'].append(results['pose_rmse'])
                results_summary['hand_rmse'].append(results['hand_rmse'])
                results_summary['overall_max_error'].append(results['overall_max_error'])
                
                print(f"  ✅ 样本 {i+1}/{num_samples} 完成")
                
            except Exception as e:
                print(f"  ❌ 样本 {i+1}/{num_samples} 失败: {e}")
                continue
                
        # 统计结果
        print(f"\n📊 批量验证统计结果:")
        for key, values in results_summary.items():
            if values:
                print(f"  {key}: mean={np.mean(values):.6f}, std={np.std(values):.6f}, max={np.max(values):.6f}")
                
        return results_summary


def main():
    parser = argparse.ArgumentParser(description="离线策略验证脚本")
    parser.add_argument("-mp", "--model_path", required=True, help="模型路径")
    parser.add_argument("-ckpt", "--checkpoint", type=int, default=600, help="检查点编号")
    parser.add_argument("-d", "--dataset", default="/home/gray/Project/DexUMI/data/dataset_0909_camera1_0918_rotvec.zarr", help="数据集路径")
    parser.add_argument("-e", "--episode", type=int, default=0, help="Episode索引")
    parser.add_argument("-s", "--step", type=int, default=0, help="时间步索引")
    parser.add_argument("-o", "--output", help="可视化结果保存路径")
    parser.add_argument("--batch", type=int, help="批量验证样本数量")
    
    args = parser.parse_args()
    
    # 创建调试器
    debugger = OfflinePolicyDebugger(
        model_path=args.model_path,
        ckpt=args.checkpoint,
        dataset_path=args.dataset,
    )
    
    if args.batch:
        # 批量验证
        debugger.batch_validation(num_samples=args.batch)
    else:
        # 单步验证
        results = debugger.single_step_prediction(
            episode_idx=args.episode,
            step_idx=args.step
        )
        
        # 可视化结果 - 同时生成三种分析
        debugger.visualize_results(results, save_path=args.output)  # 图像预处理对比
        debugger.visualize_dual_coordinate_analysis(results, save_path=args.output)  # 双重坐标系分析
        debugger.visualize_trajectory_analysis(results, save_path=args.output)  # 传统16步轨迹分析


if __name__ == "__main__":
    main()