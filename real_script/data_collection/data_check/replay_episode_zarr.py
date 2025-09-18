#!/usr/bin/env python3
"""
Episode Data Replay Script for DexUMI System

This script replays collected episode data to validate:
1. TCP pose trajectories
2. Hand action sequences
3. Data quality and consistency

It reads zarr datasets and replays the data through the same HTTP interface
used during data collection and inference.
"""

import os
import sys
import time
import argparse
import numpy as np
import zarr
import requests
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import scipy.spatial.transform as st
from pathlib import Path

# Add parent directories to path for imports (removed dependency on complex HTTP clients)
# sys.path.append(str(Path(__file__).parent.parent.parent))
# from dexumi.real_env.common.http_client import HTTPRobotClient, HTTPHandClient


class EpisodeDataReplay:
    """Class for replaying episode data from zarr datasets"""

    def __init__(self, zarr_path: str, base_url: str = "http://127.0.0.1:5000"):
        """
        Initialize the replay system

        Args:
            zarr_path: Path to zarr dataset
            base_url: Base URL for HTTP robot/hand control
        """
        self.zarr_path = zarr_path
        self.base_url = base_url

        # Initialize HTTP clients (removed dependency on complex clients)
        # self.robot_client = HTTPRobotClient(base_url=base_url)
        # self.hand_client = HTTPHandClient(base_url=base_url)

        # Load zarr dataset
        self.dataset = zarr.open(zarr_path, mode='r')
        self.episodes = list(self.dataset.group_keys())

        print(f"Loaded dataset: {zarr_path}")
        print(f"Found {len(self.episodes)} episodes: {self.episodes}")

    def get_episode_info(self, episode_name: str) -> Dict:
        """Get information about an episode"""
        if episode_name not in self.episodes:
            raise ValueError(f"Episode {episode_name} not found. Available: {self.episodes}")

        episode = self.dataset[episode_name]
        info = {
            'episode_name': episode_name,
            'pose_shape': episode['pose'].shape if 'pose' in episode else None,
            'hand_action_shape': episode['hand_action'].shape if 'hand_action' in episode else None,
            'proprioception_shape': episode['proprioception'].shape if 'proprioception' in episode else None,
            'fsr_shape': episode['fsr'].shape if 'fsr' in episode else None,
            'cameras': []
        }

        # Check camera data
        for key in episode.group_keys():
            if key.startswith('camera_'):
                cam_info = {
                    'camera_id': key,
                    'rgb_shape': episode[key]['rgb'].shape if 'rgb' in episode[key] else None
                }
                info['cameras'].append(cam_info)

        return info

    def load_episode_data(self, episode_name: str) -> Dict:
        """Load all data for an episode"""
        if episode_name not in self.episodes:
            raise ValueError(f"Episode {episode_name} not found")

        episode = self.dataset[episode_name]

        # Load core data
        data = {
            'pose': np.array(episode['pose'][:]) if 'pose' in episode else None,
            'hand_action': np.array(episode['hand_action'][:]) if 'hand_action' in episode else None,
            'proprioception': np.array(episode['proprioception'][:]) if 'proprioception' in episode else None,
            'fsr': np.array(episode['fsr'][:]) if 'fsr' in episode else None,
            'cameras': {}
        }

        # Load camera data
        for key in episode.group_keys():
            if key.startswith('camera_'):
                data['cameras'][key] = {
                    'rgb': np.array(episode[key]['rgb'][:]) if 'rgb' in episode[key] else None
                }

        return data

    def validate_data_consistency(self, data: Dict) -> Dict:
        """Validate data consistency and return statistics"""
        stats = {
            'consistent': True,
            'lengths': {},
            'issues': []
        }

        # Check lengths
        if data['pose'] is not None:
            stats['lengths']['pose'] = len(data['pose'])
        if data['hand_action'] is not None:
            stats['lengths']['hand_action'] = len(data['hand_action'])
        if data['proprioception'] is not None:
            stats['lengths']['proprioception'] = len(data['proprioception'])
        if data['fsr'] is not None:
            stats['lengths']['fsr'] = len(data['fsr'])

        for cam_name, cam_data in data['cameras'].items():
            if cam_data['rgb'] is not None:
                stats['lengths'][f'{cam_name}_rgb'] = len(cam_data['rgb'])

        # Check consistency
        lengths = list(stats['lengths'].values())
        if len(set(lengths)) > 1:
            stats['consistent'] = False
            stats['issues'].append(f"Inconsistent lengths: {stats['lengths']}")

        return stats

    def analyze_trajectory_quality(self, pose_data: np.ndarray) -> Dict:
        """Analyze pose trajectory quality"""
        if pose_data is None or len(pose_data) == 0:
            return {'error': 'No pose data available'}

        # Extract position and rotation
        positions = pose_data[:, :3]  # xyz
        rotations = pose_data[:, 3:]  # euler angles

        # Calculate velocities (approximate)
        dt = 0.05  # Assuming 20Hz data collection
        pos_velocities = np.diff(positions, axis=0) / dt
        rot_velocities = np.diff(rotations, axis=0) / dt

        # Calculate statistics
        analysis = {
            'total_frames': len(pose_data),
            'duration_estimate': len(pose_data) * dt,
            'position_stats': {
                'mean': np.mean(positions, axis=0),
                'std': np.std(positions, axis=0),
                'range': np.ptp(positions, axis=0),  # peak-to-peak
                'max_velocity': np.max(np.linalg.norm(pos_velocities, axis=1)) if len(pos_velocities) > 0 else 0
            },
            'rotation_stats': {
                'mean': np.mean(rotations, axis=0),
                'std': np.std(rotations, axis=0),
                'range': np.ptp(rotations, axis=0),
                'max_angular_velocity': np.max(np.linalg.norm(rot_velocities, axis=1)) if len(rot_velocities) > 0 else 0
            }
        }

        return analysis

    def analyze_hand_trajectory_quality(self, hand_data: np.ndarray) -> Dict:
        """Analyze hand action trajectory quality"""
        if hand_data is None or len(hand_data) == 0:
            return {'error': 'No hand action data available'}

        # Calculate velocities
        dt = 0.05  # Assuming 20Hz data collection
        velocities = np.diff(hand_data, axis=0) / dt

        analysis = {
            'total_frames': len(hand_data),
            'duration_estimate': len(hand_data) * dt,
            'joint_count': hand_data.shape[1],
            'joint_stats': {
                'mean': np.mean(hand_data, axis=0),
                'std': np.std(hand_data, axis=0),
                'range': np.ptp(hand_data, axis=0),
                'max_velocity': np.max(np.abs(velocities), axis=0) if len(velocities) > 0 else np.zeros(hand_data.shape[1])
            }
        }

        return analysis

    def visualize_trajectories(self, data: Dict, save_path: Optional[str] = None):
        """Visualize pose and hand trajectories"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot pose trajectory
        if data['pose'] is not None:
            pose_data = data['pose']
            positions = pose_data[:, :3]
            rotations = pose_data[:, 3:]

            # Plot position trajectory
            axes[0, 0].plot(positions[:, 0], label='X', alpha=0.7)
            axes[0, 0].plot(positions[:, 1], label='Y', alpha=0.7)
            axes[0, 0].plot(positions[:, 2], label='Z', alpha=0.7)
            axes[0, 0].set_title('TCP Position Trajectory')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Plot rotation trajectory
            axes[0, 1].plot(rotations[:, 0], label='Roll', alpha=0.7)
            axes[0, 1].plot(rotations[:, 1], label='Pitch', alpha=0.7)
            axes[0, 1].plot(rotations[:, 2], label='Yaw', alpha=0.7)
            axes[0, 1].set_title('TCP Rotation Trajectory')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Rotation (rad)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Plot hand trajectory
        if data['hand_action'] is not None:
            hand_data = data['hand_action']

            # Plot all joints
            for i in range(min(hand_data.shape[1], 6)):  # Show first 6 joints
                axes[1, 0].plot(hand_data[:, i], label=f'Joint {i}', alpha=0.7)
            axes[1, 0].set_title('Hand Joint Positions (First 6 joints)')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Position')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Plot remaining joints if any
            if hand_data.shape[1] > 6:
                for i in range(6, min(hand_data.shape[1], 12)):
                    axes[1, 1].plot(hand_data[:, i], label=f'Joint {i}', alpha=0.7)
                axes[1, 1].set_title('Hand Joint Positions (Joints 6-11)')
                axes[1, 1].set_xlabel('Frame')
                axes[1, 1].set_ylabel('Position')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plots saved to: {save_path}")
        else:
            plt.show()

    def convert_pose_to_quaternion(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Convert pose data from zarr format (xyz + rotvec) to robot control format (xyz + quaternion)

        Args:
            pose_data: [N, 6] array with [x, y, z, rx, ry, rz] (rotation vector)

        Returns:
            [N, 7] array with [x, y, z, qx, qy, qz, qw] (quaternion in XYZW format)
        """
        if pose_data.shape[1] != 6:
            raise ValueError(f"Expected 6D pose data, got shape {pose_data.shape}")

        converted_poses = np.zeros((len(pose_data), 7))
        converted_poses[:, :3] = pose_data[:, :3]  # Copy position

        # Convert rotation vectors to quaternions
        for i in range(len(pose_data)):
            rotvec = pose_data[i, 3:]  # [rx, ry, rz]
            quat = st.Rotation.from_rotvec(rotvec).as_quat()  # Returns [x, y, z, w]
            converted_poses[i, 3:] = quat

        return converted_poses

    def send_pose_command(self, pose: np.ndarray) -> bool:
        """Send pose command to robot via HTTP"""
        try:
            data = {"arr": pose.tolist()}
            response = requests.post(f"{self.base_url}/pose", json=data, timeout=1.0)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send pose command: {e}")
            return False

    def send_hand_command(self, hand_action: np.ndarray) -> bool:
        """Send hand action command via HTTP"""
        try:
            data = {"arr": hand_action.tolist()}
            response = requests.post(f"{self.base_url}/hand_pose", json=data, timeout=1.0)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send hand command: {e}")
            return False

    def replay_episode(self, episode_name: str, playback_speed: float = 1.0,
                      start_frame: int = 0, end_frame: Optional[int] = None,
                      send_commands: bool = False, dry_run: bool = True):
        """
        Replay an episode

        Args:
            episode_name: Name of episode to replay
            playback_speed: Speed multiplier (1.0 = real-time)
            start_frame: Starting frame index
            end_frame: Ending frame index (None = all frames)
            send_commands: Whether to actually send commands to robot/hand
            dry_run: If True, just print commands without sending
        """
        print(f"\n{'='*60}")
        print(f"Replaying Episode: {episode_name}")
        print(f"Playback Speed: {playback_speed}x")
        print(f"Send Commands: {send_commands} (Dry Run: {dry_run})")
        print(f"{'='*60}\n")

        # Load episode data
        data = self.load_episode_data(episode_name)

        # Validate data
        stats = self.validate_data_consistency(data)
        if not stats['consistent']:
            print("WARNING: Data consistency issues found:")
            for issue in stats['issues']:
                print(f"  - {issue}")

        # Convert pose format from rotvec to quaternion for robot control
        if data['pose'] is not None:
            print("Converting pose format: rotation vectors -> quaternions")
            data['pose'] = self.convert_pose_to_quaternion(data['pose'])
            print("✅ Pose format converted")

        # Determine frame range
        if data['pose'] is not None:
            total_frames = len(data['pose'])
        elif data['hand_action'] is not None:
            total_frames = len(data['hand_action'])
        else:
            print("ERROR: No pose or hand action data found")
            return

        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)

        print(f"Replaying frames {start_frame} to {end_frame-1} ({end_frame-start_frame} frames)")

        # Calculate timing
        dt = 0.05  # Assume 20Hz data collection
        frame_interval = dt / playback_speed

        # Replay loop
        start_time = time.time()

        for frame_idx in range(start_frame, end_frame):
            frame_start = time.time()

            # Get current frame data
            current_pose = data['pose'][frame_idx] if data['pose'] is not None else None
            current_hand = data['hand_action'][frame_idx] if data['hand_action'] is not None else None

            # Print current state
            print(f"Frame {frame_idx:4d}/{end_frame-1:4d} | ", end="")

            if current_pose is not None:
                pos = current_pose[:3]
                rot = current_pose[3:]  # Now quaternion [qx, qy, qz, qw]
                print(f"TCP: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] ", end="")
                print(f"Quat: [{rot[0]:5.3f}, {rot[1]:5.3f}, {rot[2]:5.3f}, {rot[3]:5.3f}] | ", end="")

            if current_hand is not None:
                # Show first few joint values
                hand_str = "[" + ", ".join([f"{x:6.3f}" for x in current_hand[:4]]) + "...]"
                print(f"Hand: {hand_str} | ", end="")

            # Send commands if requested
            if send_commands and not dry_run:
                success = True

                # Send robot pose command
                if current_pose is not None:
                    # current_pose is now in correct format (xyz + quaternion) after conversion
                    if not self.send_pose_command(current_pose):
                        success = False

                # Send hand command
                if current_hand is not None:
                    if not self.send_hand_command(current_hand):
                        success = False

                if success:
                    print("✓ Commands sent")
                else:
                    print("✗ Command failed")
            else:
                print("○ Dry run" if dry_run else "○ Commands disabled")

            # Wait for next frame
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_time = time.time() - start_time
        expected_time = (end_frame - start_frame) * frame_interval
        print(f"\nReplay completed!")
        print(f"Total time: {total_time:.2f}s (expected: {expected_time:.2f}s)")
        print(f"Average frame rate: {(end_frame - start_frame) / total_time:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Replay episode data for validation")
    parser.add_argument("zarr_path", help="Path to zarr dataset")
    parser.add_argument("--episode", "-e", type=str, help="Episode name to replay (e.g., episode_0)")
    parser.add_argument("--list", "-l", action="store_true", help="List available episodes")
    parser.add_argument("--info", "-i", action="store_true", help="Show episode information")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze trajectory quality")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize trajectories")
    parser.add_argument("--replay", "-r", action="store_true", help="Replay episode")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index")
    parser.add_argument("--send-commands", action="store_true", help="Actually send commands to robot/hand")
    parser.add_argument("--live", action="store_true", help="Disable dry-run mode (use with --send-commands)")
    parser.add_argument("--save-plots", type=str, help="Save plots to file path")

    args = parser.parse_args()

    # Check if zarr path exists
    if not os.path.exists(args.zarr_path):
        print(f"Error: Dataset path '{args.zarr_path}' does not exist")
        return

    # Initialize replay system
    try:
        replay_system = EpisodeDataReplay(args.zarr_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # List episodes if requested
    if args.list:
        print(f"\nAvailable episodes in {args.zarr_path}:")
        for ep in replay_system.episodes:
            print(f"  - {ep}")
        return

    # Require episode selection for other operations
    if not args.episode:
        print("Error: Please specify an episode with --episode")
        print("Use --list to see available episodes")
        return

    # Show episode information
    if args.info or args.analyze or args.visualize or args.replay:
        try:
            info = replay_system.get_episode_info(args.episode)
            print(f"\nEpisode Information: {args.episode}")
            print(f"  Pose shape: {info['pose_shape']}")
            print(f"  Hand action shape: {info['hand_action_shape']}")
            print(f"  Proprioception shape: {info['proprioception_shape']}")
            print(f"  FSR shape: {info['fsr_shape']}")
            print(f"  Cameras: {len(info['cameras'])}")
            for cam in info['cameras']:
                print(f"    - {cam['camera_id']}: {cam['rgb_shape']}")
        except Exception as e:
            print(f"Error getting episode info: {e}")
            return

    # Analyze trajectory quality
    if args.analyze:
        print(f"\nAnalyzing trajectory quality for {args.episode}...")
        try:
            data = replay_system.load_episode_data(args.episode)

            # Analyze pose trajectory
            if data['pose'] is not None:
                pose_analysis = replay_system.analyze_trajectory_quality(data['pose'])
                print(f"\nPose Trajectory Analysis:")
                print(f"  Total frames: {pose_analysis['total_frames']}")
                print(f"  Duration estimate: {pose_analysis['duration_estimate']:.2f}s")
                print(f"  Position range: {pose_analysis['position_stats']['range']}")
                print(f"  Max velocity: {pose_analysis['position_stats']['max_velocity']:.3f} m/s")
                print(f"  Max angular velocity: {pose_analysis['rotation_stats']['max_angular_velocity']:.3f} rad/s")

            # Analyze hand trajectory
            if data['hand_action'] is not None:
                hand_analysis = replay_system.analyze_hand_trajectory_quality(data['hand_action'])
                print(f"\nHand Trajectory Analysis:")
                print(f"  Total frames: {hand_analysis['total_frames']}")
                print(f"  Joint count: {hand_analysis['joint_count']}")
                print(f"  Joint ranges: {hand_analysis['joint_stats']['range']}")
                max_vel = hand_analysis['joint_stats']['max_velocity']
                print(f"  Max joint velocities: {max_vel[:4]}... (first 4 joints)")

        except Exception as e:
            print(f"Error analyzing trajectories: {e}")

    # Visualize trajectories
    if args.visualize:
        print(f"\nVisualizing trajectories for {args.episode}...")
        try:
            data = replay_system.load_episode_data(args.episode)
            replay_system.visualize_trajectories(data, save_path=args.save_plots)
        except Exception as e:
            print(f"Error visualizing trajectories: {e}")

    # Replay episode
    if args.replay:
        print(f"\nReplaying {args.episode}...")
        try:
            replay_system.replay_episode(
                episode_name=args.episode,
                playback_speed=args.speed,
                start_frame=args.start,
                end_frame=args.end,
                send_commands=args.send_commands,
                dry_run=not args.live
            )
        except Exception as e:
            print(f"Error replaying episode: {e}")


if __name__ == "__main__":
    main()