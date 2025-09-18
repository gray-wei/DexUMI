#!/usr/bin/env python3
"""
完整Episode数据回放脚本 - 同时控制机械臂和机械手

功能:
1. 加载episode中的pose.pkl和hand_action.pkl轨迹数据
2. 通过Franka+XHand HTTP接口同步回放轨迹
3. 提供完整的可视化和统计分析
4. 支持精度验证和数据质量检查

使用方法:
python replay_episode_pkl.py --episode_path /path/to/episode_dir [--options]
"""

"""
# 基本分析 (推荐首次使用)
python replay_episode_pkl.py --episode_path /path/to/episode_0 --analyze_only --visualize

# 干跑回放 (不发送实际命令)
python replay_episode_pkl.py --episode_path /path/to/episode_0

# 实际同步回放 (危险操作！)
python replay_episode_pkl.py --episode_path /path/to/episode_0 --send_commands --frequency 10.0

# 完整精度检查
python replay_episode_pkl.py --episode_path /path/to/episode_0 --check_accuracy

# 保存完整分析结果
python replay_episode_pkl.py --episode_path /path/to/episode_0 --analyze_only --save_analysis complete_analysis.json
"""

import argparse
import pickle
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Optional, Tuple
import json
import os
import sys
import threading
from queue import Queue

class CompleteEpisodeReplayer:
    """完整Episode轨迹回放器 - 机械臂+机械手"""

    def __init__(self, server_url: str = "http://127.0.0.1:5000"):
        self.server_url = server_url.rstrip('/')

        # 数据存储
        self.poses = None
        self.hand_actions = None
        self.timestamps = None

        # 控制状态
        self.current_index = 0
        self.is_playing = False
        self.start_time = None

        # 命令队列 (用于并发控制)
        self.pose_command_queue = Queue()
        self.hand_command_queue = Queue()

    def load_episode_data(self, episode_path: str) -> bool:
        """加载完整episode数据"""
        try:
            # 加载pose数据
            pose_file = os.path.join(episode_path, "pose.pkl")
            if not os.path.exists(pose_file):
                print(f"错误: pose.pkl文件不存在: {pose_file}")
                return False

            with open(pose_file, 'rb') as f:
                self.poses = pickle.load(f)

            # 加载hand_action数据
            hand_file = os.path.join(episode_path, "hand_action.pkl")
            if not os.path.exists(hand_file):
                print(f"错误: hand_action.pkl文件不存在: {hand_file}")
                return False

            with open(hand_file, 'rb') as f:
                self.hand_actions = pickle.load(f)

            # 加载时间戳数据
            timestamp_file = os.path.join(episode_path, "timestamps.pkl")
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'rb') as f:
                    timestamp_data = pickle.load(f)
                    if isinstance(timestamp_data, dict):
                        self.timestamps = timestamp_data.get('main_timestamps', None)
                    else:
                        self.timestamps = timestamp_data

            # 如果没有时间戳，创建等间隔时间戳(假设20Hz)
            if self.timestamps is None:
                print("警告: 未找到时间戳数据，使用默认20Hz间隔")
                self.timestamps = np.arange(len(self.poses)) * 0.05  # 20Hz = 50ms间隔

            # 验证数据长度一致性
            if len(self.poses) != len(self.hand_actions):
                print(f"警告: pose和hand_action长度不一致: {len(self.poses)} vs {len(self.hand_actions)}")
                min_len = min(len(self.poses), len(self.hand_actions))
                self.poses = self.poses[:min_len]
                self.hand_actions = self.hand_actions[:min_len]
                self.timestamps = self.timestamps[:min_len]

            print(f"✓ 成功加载完整episode数据:")
            print(f"  - Pose数据: {self.poses.shape}")
            print(f"  - Hand action数据: {self.hand_actions.shape}")
            print(f"  - 时间戳数量: {len(self.timestamps)}")

            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def check_server_connection(self) -> bool:
        """检查服务器连接"""
        try:
            # 检查机械臂连接
            response = requests.post(f"{self.server_url}/getpos", timeout=2.0)
            if response.status_code != 200:
                print(f"机械臂服务器响应错误: {response.status_code}")
                return False

            # 检查机械手连接
            response = requests.post(f"{self.server_url}/get_handangle", timeout=2.0)
            if response.status_code != 200:
                print(f"机械手服务器响应错误: {response.status_code}")
                return False

            print("✓ Franka+XHand服务器连接正常")
            return True

        except Exception as e:
            print(f"服务器连接失败: {e}")
            return False

    def get_current_state(self) -> Optional[Dict]:
        """获取当前机械臂和机械手状态"""
        try:
            # 获取机械臂状态
            response = requests.post(f"{self.server_url}/getstate", timeout=1.0)
            if response.status_code == 200:
                data = response.json()
                return {
                    'pose': np.array(data["pose"]),
                    'hand_angles': np.array(data["gripper_pos"])
                }
            return None
        except Exception as e:
            print(f"获取当前状态失败: {e}")
            return None

    def send_pose_command(self, pose: np.ndarray) -> bool:
        """发送pose命令到机械臂"""
        try:
            data = {"arr": pose.tolist()}
            response = requests.post(f"{self.server_url}/pose",
                                   json=data, timeout=1.0)
            return response.status_code == 200
        except Exception as e:
            print(f"发送pose命令失败: {e}")
            return False

    def send_hand_command(self, hand_angles: np.ndarray) -> bool:
        """发送hand命令到机械手"""
        try:
            data = {"arr": hand_angles.tolist()}
            response = requests.post(f"{self.server_url}/hand_pose",
                                   json=data, timeout=1.0)
            return response.status_code == 200
        except Exception as e:
            print(f"发送hand命令失败: {e}")
            return False

    def analyze_complete_trajectory(self) -> Dict:
        """分析完整轨迹统计信息"""
        if self.poses is None or self.hand_actions is None:
            return {}

        # 分析pose轨迹
        positions = self.poses[:, :3]
        quaternions = self.poses[:, 3:7]

        # 位置统计
        pos_stats = {
            'min': np.min(positions, axis=0),
            'max': np.max(positions, axis=0),
            'mean': np.mean(positions, axis=0),
            'std': np.std(positions, axis=0),
            'range': np.max(positions, axis=0) - np.min(positions, axis=0)
        }

        # 速度统计（数值微分）
        if len(self.timestamps) > 1:
            dt = np.diff(self.timestamps)
            velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
            vel_magnitudes = np.linalg.norm(velocities, axis=1)

            vel_stats = {
                'max_velocity': np.max(vel_magnitudes),
                'mean_velocity': np.mean(vel_magnitudes),
                'std_velocity': np.std(vel_magnitudes)
            }
        else:
            vel_stats = {'max_velocity': 0, 'mean_velocity': 0, 'std_velocity': 0}

        # 旋转变化统计
        rotation_changes = []
        for i in range(1, len(quaternions)):
            r1 = R.from_quat(quaternions[i-1])
            r2 = R.from_quat(quaternions[i])
            angle_diff = (r1.inv() * r2).magnitude()
            rotation_changes.append(angle_diff)

        rot_stats = {
            'max_rotation_change': np.max(rotation_changes) if rotation_changes else 0,
            'mean_rotation_change': np.mean(rotation_changes) if rotation_changes else 0
        }

        # 分析hand轨迹
        hand_stats = {
            'joint_count': self.hand_actions.shape[1],
            'min': np.min(self.hand_actions, axis=0),
            'max': np.max(self.hand_actions, axis=0),
            'mean': np.mean(self.hand_actions, axis=0),
            'std': np.std(self.hand_actions, axis=0),
            'range': np.max(self.hand_actions, axis=0) - np.min(self.hand_actions, axis=0)
        }

        # hand速度统计
        if len(self.timestamps) > 1:
            hand_velocities = np.diff(self.hand_actions, axis=0) / dt[:, np.newaxis]
            hand_vel_stats = {
                'max_joint_velocities': np.max(np.abs(hand_velocities), axis=0),
                'mean_joint_velocities': np.mean(np.abs(hand_velocities), axis=0)
            }
        else:
            hand_vel_stats = {
                'max_joint_velocities': np.zeros(self.hand_actions.shape[1]),
                'mean_joint_velocities': np.zeros(self.hand_actions.shape[1])
            }

        return {
            'total_points': len(self.poses),
            'duration': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0,
            'pose_stats': {
                'position_stats': pos_stats,
                'velocity_stats': vel_stats,
                'rotation_stats': rot_stats
            },
            'hand_stats': {
                'joint_stats': hand_stats,
                'velocity_stats': hand_vel_stats
            }
        }

    def visualize_complete_trajectory(self, save_path: Optional[str] = None):
        """可视化完整轨迹"""
        if self.poses is None or self.hand_actions is None:
            print("错误: 没有加载轨迹数据")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('完整Episode轨迹分析', fontsize=16)

        positions = self.poses[:, :3]
        quaternions = self.poses[:, 3:7]

        # 机械臂轨迹分析
        # 1. 3D轨迹
        ax1 = axes[0, 0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, label='轨迹')
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=50, label='起点')
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, label='终点')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('机械臂 X-Y轨迹')
        ax1.legend()
        ax1.grid(True)

        # 2. Z轴变化
        ax2 = axes[0, 1]
        ax2.plot(self.timestamps, positions[:, 2], 'r-', linewidth=2)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('机械臂 Z轴高度变化')
        ax2.grid(True)

        # 3. 位置变化
        ax3 = axes[1, 0]
        ax3.plot(self.timestamps, positions[:, 0], 'r-', label='X', alpha=0.7)
        ax3.plot(self.timestamps, positions[:, 1], 'g-', label='Y', alpha=0.7)
        ax3.plot(self.timestamps, positions[:, 2], 'b-', label='Z', alpha=0.7)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('位置 (m)')
        ax3.set_title('机械臂位置随时间变化')
        ax3.legend()
        ax3.grid(True)

        # 4. 旋转角度变化
        ax4 = axes[1, 1]
        euler_angles = np.array([R.from_quat(q).as_euler('xyz') for q in quaternions])
        ax4.plot(self.timestamps, np.degrees(euler_angles[:, 0]), 'r-', label='Roll', alpha=0.7)
        ax4.plot(self.timestamps, np.degrees(euler_angles[:, 1]), 'g-', label='Pitch', alpha=0.7)
        ax4.plot(self.timestamps, np.degrees(euler_angles[:, 2]), 'b-', label='Yaw', alpha=0.7)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('角度 (度)')
        ax4.set_title('机械臂欧拉角变化')
        ax4.legend()
        ax4.grid(True)

        # 机械手轨迹分析
        # 5. 手部关节变化 (前6个关节)
        ax5 = axes[2, 0]
        for i in range(min(6, self.hand_actions.shape[1])):
            ax5.plot(self.timestamps, self.hand_actions[:, i], label=f'关节 {i}', alpha=0.7)
        ax5.set_xlabel('时间 (s)')
        ax5.set_ylabel('关节角度')
        ax5.set_title('机械手关节角度变化 (前6个关节)')
        ax5.legend()
        ax5.grid(True)

        # 6. 手部关节变化 (后6个关节)
        ax6 = axes[2, 1]
        start_idx = min(6, self.hand_actions.shape[1])
        for i in range(start_idx, min(12, self.hand_actions.shape[1])):
            ax6.plot(self.timestamps, self.hand_actions[:, i], label=f'关节 {i}', alpha=0.7)
        ax6.set_xlabel('时间 (s)')
        ax6.set_ylabel('关节角度')
        ax6.set_title('机械手关节角度变化 (后6个关节)')
        ax6.legend()
        ax6.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 完整轨迹图保存到: {save_path}")
        else:
            plt.show()

    def replay_complete_trajectory(self, frequency: float = 20.0, start_index: int = 0,
                                 end_index: Optional[int] = None, interactive: bool = True,
                                 send_commands: bool = False):
        """回放完整轨迹 - 机械臂+机械手同步"""
        if self.poses is None or self.hand_actions is None:
            print("错误: 没有加载完整轨迹数据")
            return False

        if send_commands and not self.check_server_connection():
            print("错误: 无法连接到服务器")
            return False

        end_index = end_index or len(self.poses)
        end_index = min(end_index, len(self.poses))

        print(f"\n开始回放完整轨迹:")
        print(f"  起始索引: {start_index}")
        print(f"  结束索引: {end_index}")
        print(f"  回放频率: {frequency} Hz")
        print(f"  总点数: {end_index - start_index}")
        print(f"  发送命令: {send_commands}")

        self.current_index = start_index
        self.is_playing = True
        dt = 1.0 / frequency

        # 记录回放统计
        successful_frames = 0
        pose_failures = 0
        hand_failures = 0

        try:
            while self.current_index < end_index and self.is_playing:
                loop_start = time.time()

                # 获取当前帧数据
                pose = self.poses[self.current_index]
                hand_angles = self.hand_actions[self.current_index]

                # 显示进度
                progress = (self.current_index - start_index) / (end_index - start_index) * 100
                print(f"\r进度: {progress:.1f}% | 索引: {self.current_index}/{end_index} | "
                      f"Pose: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] | "
                      f"Hand: [{hand_angles[0]:.3f}, {hand_angles[1]:.3f}, ...]", end='')

                if send_commands:
                    # 并发发送命令以提高效率
                    pose_success = self.send_pose_command(pose)
                    hand_success = self.send_hand_command(hand_angles)

                    if pose_success and hand_success:
                        successful_frames += 1
                    if not pose_success:
                        pose_failures += 1
                    if not hand_success:
                        hand_failures += 1

                self.current_index += 1

                # 控制频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)

            print(f"\n✓ 完整轨迹回放完成!")

            if send_commands:
                total_frames = end_index - start_index
                success_rate = (successful_frames / total_frames) * 100
                print(f"\n回放统计:")
                print(f"  - 成功帧数: {successful_frames}/{total_frames} ({success_rate:.1f}%)")
                print(f"  - Pose失败: {pose_failures}")
                print(f"  - Hand失败: {hand_failures}")

        except KeyboardInterrupt:
            print(f"\n用户中断回放")
            return False
        except Exception as e:
            print(f"\n回放过程中出错: {e}")
            return False

        return True

    def compare_with_actual_complete(self, tolerance_pose: float = 0.01,
                                   tolerance_hand: float = 0.1) -> Dict:
        """比较期望状态和实际状态"""
        if self.poses is None or self.hand_actions is None:
            return {}

        print("开始完整精度验证...")
        errors = []

        sample_indices = range(0, len(self.poses), 10)  # 每10个点采样一次

        for i, idx in enumerate(sample_indices):
            expected_pose = self.poses[idx]
            expected_hand = self.hand_actions[idx]

            # 发送命令
            pose_success = self.send_pose_command(expected_pose)
            hand_success = self.send_hand_command(expected_hand)

            if pose_success and hand_success:
                time.sleep(0.1)  # 等待运动完成

                # 获取实际状态
                actual_state = self.get_current_state()
                if actual_state is not None:
                    actual_pose = actual_state['pose']
                    actual_hand = actual_state['hand_angles']

                    # 计算位置误差
                    pos_error = np.linalg.norm(expected_pose[:3] - actual_pose[:3])

                    # 计算旋转误差
                    expected_quat = expected_pose[3:7]
                    actual_quat = actual_pose[3:7]
                    rot_error = np.abs(1 - np.abs(np.dot(expected_quat, actual_quat)))

                    # 计算手部误差
                    hand_error = np.linalg.norm(expected_hand - actual_hand)

                    errors.append({
                        'index': idx,
                        'position_error': pos_error,
                        'rotation_error': rot_error,
                        'hand_error': hand_error,
                        'expected_pose': expected_pose.copy(),
                        'actual_pose': actual_pose.copy(),
                        'expected_hand': expected_hand.copy(),
                        'actual_hand': actual_hand.copy()
                    })

                    print(f"  样本 {i}: 位置误差={pos_error:.4f}m, 旋转误差={rot_error:.4f}, 手部误差={hand_error:.4f}")

        if errors:
            pos_errors = [e['position_error'] for e in errors]
            rot_errors = [e['rotation_error'] for e in errors]
            hand_errors = [e['hand_error'] for e in errors]

            accuracy_stats = {
                'sample_count': len(errors),
                'pose_accuracy': {
                    'position_error_mean': np.mean(pos_errors),
                    'position_error_max': np.max(pos_errors),
                    'position_error_std': np.std(pos_errors),
                    'rotation_error_mean': np.mean(rot_errors),
                    'rotation_error_max': np.max(rot_errors),
                    'position_accuracy_within_tolerance': np.mean(np.array(pos_errors) < tolerance_pose) * 100
                },
                'hand_accuracy': {
                    'hand_error_mean': np.mean(hand_errors),
                    'hand_error_max': np.max(hand_errors),
                    'hand_error_std': np.std(hand_errors),
                    'hand_accuracy_within_tolerance': np.mean(np.array(hand_errors) < tolerance_hand) * 100
                }
            }

            print(f"\n精度统计:")
            print(f"  机械臂精度:")
            print(f"    平均位置误差: {accuracy_stats['pose_accuracy']['position_error_mean']:.4f} m")
            print(f"    最大位置误差: {accuracy_stats['pose_accuracy']['position_error_max']:.4f} m")
            print(f"    在{tolerance_pose}m容差内的比例: {accuracy_stats['pose_accuracy']['position_accuracy_within_tolerance']:.1f}%")

            print(f"  机械手精度:")
            print(f"    平均关节误差: {accuracy_stats['hand_accuracy']['hand_error_mean']:.4f}")
            print(f"    最大关节误差: {accuracy_stats['hand_accuracy']['hand_error_max']:.4f}")
            print(f"    在{tolerance_hand}容差内的比例: {accuracy_stats['hand_accuracy']['hand_accuracy_within_tolerance']:.1f}%")

            return accuracy_stats

        return {}


def main():
    parser = argparse.ArgumentParser(description="回放完整Episode轨迹数据 (机械臂+机械手)")
    parser.add_argument("--episode_path", type=str, required=True,
                       help="Episode数据目录路径")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000",
                       help="服务器URL")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="回放频率 (Hz)")
    parser.add_argument("--start_index", type=int, default=0,
                       help="起始索引")
    parser.add_argument("--end_index", type=int, default=None,
                       help="结束索引")
    parser.add_argument("--analyze_only", action="store_true",
                       help="仅分析轨迹，不执行回放")
    parser.add_argument("--visualize", action="store_true",
                       help="显示轨迹可视化图")
    parser.add_argument("--check_accuracy", action="store_true",
                       help="检查精度")
    parser.add_argument("--send_commands", action="store_true",
                       help="实际发送控制命令")
    parser.add_argument("--save_analysis", type=str, default=None,
                       help="保存分析结果到文件")

    args = parser.parse_args()

    # 验证episode路径
    if not os.path.exists(args.episode_path):
        print(f"错误: Episode路径不存在: {args.episode_path}")
        sys.exit(1)

    # 创建回放器
    replayer = CompleteEpisodeReplayer(args.server_url)

    # 加载数据
    if not replayer.load_episode_data(args.episode_path):
        sys.exit(1)

    # 分析轨迹
    print("\n" + "="*60)
    print("完整轨迹分析结果:")
    print("="*60)

    analysis = replayer.analyze_complete_trajectory()
    if analysis:
        print(f"总数据点数: {analysis['total_points']}")
        print(f"轨迹持续时间: {analysis['duration']:.2f} 秒")

        # 机械臂统计
        pose_stats = analysis['pose_stats']
        pos_stats = pose_stats['position_stats']
        print(f"\n机械臂位置统计:")
        print(f"  范围: X=[{pos_stats['min'][0]:.3f}, {pos_stats['max'][0]:.3f}] m")
        print(f"       Y=[{pos_stats['min'][1]:.3f}, {pos_stats['max'][1]:.3f}] m")
        print(f"       Z=[{pos_stats['min'][2]:.3f}, {pos_stats['max'][2]:.3f}] m")

        vel_stats = pose_stats['velocity_stats']
        print(f"\n机械臂速度统计:")
        print(f"  最大速度: {vel_stats['max_velocity']:.3f} m/s")
        print(f"  平均速度: {vel_stats['mean_velocity']:.3f} m/s")

        # 机械手统计
        hand_stats = analysis['hand_stats']
        joint_stats = hand_stats['joint_stats']
        print(f"\n机械手统计:")
        print(f"  关节数量: {joint_stats['joint_count']}")
        print(f"  关节范围: {joint_stats['range'][:4]}... (前4个关节)")

        hand_vel_stats = hand_stats['velocity_stats']
        print(f"  最大关节速度: {hand_vel_stats['max_joint_velocities'][:4]}... (前4个关节)")

    # 可视化
    if args.visualize:
        save_path = None
        if args.save_analysis:
            save_path = args.save_analysis.replace('.json', '_complete_trajectory.png')
        replayer.visualize_complete_trajectory(save_path)

    # 保存分析结果
    if args.save_analysis:
        with open(args.save_analysis, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"✓ 分析结果保存到: {args.save_analysis}")

    # 如果只是分析，退出
    if args.analyze_only:
        print("\n分析完成!")
        return

    # 精度检查
    if args.check_accuracy:
        print("\n" + "="*60)
        print("完整精度检查:")
        print("="*60)
        accuracy_stats = replayer.compare_with_actual_complete()

    # 回放轨迹
    if not args.check_accuracy:  # 避免重复回放
        print("\n" + "="*60)
        print("开始完整轨迹回放:")
        print("="*60)

        if args.send_commands:
            print("警告: 将实际控制机械臂和机械手!")
            input("按Enter键开始回放...")

        success = replayer.replay_complete_trajectory(
            frequency=args.frequency,
            start_index=args.start_index,
            end_index=args.end_index,
            interactive=True,
            send_commands=args.send_commands
        )

        if success:
            print("✓ 完整轨迹回放成功完成!")
        else:
            print("✗ 完整轨迹回放失败或被中断")


if __name__ == "__main__":
    main()