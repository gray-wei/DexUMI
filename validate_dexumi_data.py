#!/usr/bin/env python3
"""
验证zarr数据格式是否符合DexUMI训练要求
"""

import zarr
import numpy as np
import sys
import os

def validate_zarr_dataset(zarr_path):
    """验证zarr数据集格式"""
    print(f"\n{'='*60}")
    print(f"验证DexUMI数据格式: {zarr_path}")
    print(f"{'='*60}\n")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        episodes = list(root.group_keys())
        
        print(f"✓ 成功打开zarr文件")
        print(f"✓ 找到 {len(episodes)} 个episodes\n")
        
        all_valid = True
        total_frames = 0
        
        # 检查每个episode
        for episode_name in episodes:
            print(f"\n检查 {episode_name}:")
            print("-" * 40)
            
            episode = root[episode_name]
            episode_valid = True
            
            # 必需的数据字段
            required_fields = {
                'pose': (None, 6),  # [T, 6] xyz + euler
                'hand_action': (None, 12),  # [T, 12] 手部关节
                'proprioception': (None, 14),  # [T, 14] 7关节位置+速度
                'fsr': (None, 3),  # [T, 3] 触觉数据
            }
            
            frame_count = None
            
            for field, expected_shape in required_fields.items():
                if field in episode:
                    data = episode[field]
                    shape = data.shape
                    
                    # 检查维度
                    if expected_shape[1] is not None and shape[1] != expected_shape[1]:
                        print(f"  ✗ {field}: 维度错误 {shape} (期望 [T, {expected_shape[1]}])")
                        episode_valid = False
                    else:
                        print(f"  ✓ {field}: {shape} dtype={data.dtype}")
                    
                    # 记录帧数
                    if frame_count is None:
                        frame_count = shape[0]
                    elif shape[0] != frame_count:
                        print(f"  ✗ {field}: 帧数不一致 {shape[0]} vs {frame_count}")
                        episode_valid = False
                else:
                    print(f"  ✗ 缺少必需字段: {field}")
                    episode_valid = False
            
            # 检查相机数据
            camera_found = False
            for key in episode.group_keys():
                if key.startswith('camera_'):
                    camera_found = True
                    cam_group = episode[key]
                    if 'rgb' in cam_group:
                        rgb_data = cam_group['rgb']
                        shape = rgb_data.shape
                        
                        # 检查图像尺寸
                        if shape[1:] == (240, 240, 3):
                            print(f"  ✓ {key}/rgb: {shape} (完美匹配DexUMI)")
                        else:
                            print(f"  ⚠ {key}/rgb: {shape} (需要resize到240x240)")
                        
                        # 检查帧数一致性
                        if shape[0] != frame_count:
                            print(f"  ✗ {key}: 帧数不一致 {shape[0]} vs {frame_count}")
                            episode_valid = False
            
            if not camera_found:
                print(f"  ✗ 没有找到相机数据")
                episode_valid = False
            
            if episode_valid:
                print(f"\n  ✓ {episode_name} 验证通过 ({frame_count} 帧)")
                total_frames += frame_count
            else:
                print(f"\n  ✗ {episode_name} 验证失败")
                all_valid = False
        
        print(f"\n{'='*60}")
        print("验证总结:")
        print(f"{'='*60}")
        
        if all_valid:
            print(f"✓ 所有数据验证通过!")
            print(f"  - Episodes: {len(episodes)}")
            print(f"  - 总帧数: {total_frames}")
            print(f"  - 平均每episode: {total_frames/len(episodes):.1f} 帧")
            
            # 计算数据统计
            print(f"\n数据统计 (用于归一化):")
            
            # 收集所有数据
            all_pose = []
            all_hand_action = []
            all_fsr = []
            
            for episode_name in episodes:
                episode = root[episode_name]
                all_pose.append(episode['pose'][:])
                all_hand_action.append(episode['hand_action'][:])
                all_fsr.append(episode['fsr'][:])
            
            all_pose = np.concatenate(all_pose)
            all_hand_action = np.concatenate(all_hand_action)
            all_fsr = np.concatenate(all_fsr)
            
            print(f"\nPose统计:")
            print(f"  - Min: {all_pose.min(axis=0).round(3)}")
            print(f"  - Max: {all_pose.max(axis=0).round(3)}")
            print(f"  - Mean: {all_pose.mean(axis=0).round(3)}")
            
            print(f"\nHand Action统计:")
            print(f"  - Min: {all_hand_action.min(axis=0).round(3)}")
            print(f"  - Max: {all_hand_action.max(axis=0).round(3)}")
            print(f"  - Range: {(all_hand_action.max(axis=0) - all_hand_action.min(axis=0)).round(3)}")
            
            print(f"\nFSR统计:")
            print(f"  - Min: {all_fsr.min(axis=0).round(3)}")
            print(f"  - Max: {all_fsr.max(axis=0).round(3)}")
            print(f"  - Mean: {all_fsr.mean(axis=0).round(3)}")
            
            return True
        else:
            print(f"✗ 数据验证失败，请检查上述问题")
            return False
            
    except Exception as e:
        print(f"✗ 验证时出错: {e}")
        return False

if __name__ == "__main__":
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_optimized.zarr"
    
    if not os.path.exists(zarr_path):
        print(f"错误: 找不到zarr文件 {zarr_path}")
        sys.exit(1)
    
    success = validate_zarr_dataset(zarr_path)
    sys.exit(0 if success else 1)