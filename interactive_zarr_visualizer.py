#!/usr/bin/env python3
"""
Interactive 3D Pose + RGB Video Synchronized Player
Visualizes zarr dataset with synchronized 3D pose trajectories and RGB video playback.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse
import os
from pathlib import Path


class InteractiveZarrVisualizer:
    def __init__(self, zarr_path, episode_id=0):
        """Initialize the visualizer with zarr dataset."""
        self.zarr_path = zarr_path
        self.episode_id = episode_id
        self.current_frame = 0
        
        # Load data
        self.load_data()
        
        # Setup GUI
        self.setup_figure()
        self.setup_widgets()
        
        # Animation control
        self.is_playing = False
        self.animation = None
        
    def load_data(self):
        """Load data from zarr file."""
        print(f"Loading episode {self.episode_id} from {self.zarr_path}")
        
        try:
            self.root = zarr.open(self.zarr_path, mode='r')
            episode_key = f'episode_{self.episode_id}'
            
            if episode_key not in self.root:
                available_episodes = [k for k in self.root.keys() if k.startswith('episode_')]
                raise ValueError(f"Episode {self.episode_id} not found. Available episodes: {available_episodes[:10]}...")
            
            self.episode = self.root[episode_key]
            
            # Load RGB data
            self.rgb_data = self.episode['camera_0']['rgb'][:]
            
            # Load pose data (6D: likely position + orientation)
            self.pose_data = self.episode['pose'][:]
            
            # Additional data for context
            self.fsr_data = self.episode['fsr'][:]
            self.hand_action = self.episode['hand_action'][:]
            
            self.total_frames = len(self.rgb_data)
            
            print(f"Loaded data:")
            print(f"  RGB shape: {self.rgb_data.shape}")
            print(f"  Pose shape: {self.pose_data.shape}")  
            print(f"  Total frames: {self.total_frames}")
            print(f"  FSR (Tactile) shape: {self.fsr_data.shape}")
            print(f"  Hand Action shape: {self.hand_action.shape}")
            
            # Extract position (first 3 dims) and orientation (last 3 dims) from pose
            self.positions = self.pose_data[:, :3]  # XYZ position
            self.orientations = self.pose_data[:, 3:6]  # Orientation (euler angles or similar)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def setup_figure(self):
        """Setup the main figure with subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Interactive Zarr Visualizer - Episode {self.episode_id}', fontsize=16)
        
        # Create subplot layout: 2x2 grid with custom sizing
        gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 2, 0.3], hspace=0.3, wspace=0.3)
        
        # RGB video display (top left)
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_rgb.set_title('RGB Video')
        self.ax_rgb.axis('off')
        
        # 3D pose trajectory (top right) 
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_3d.set_title('3D Pose Trajectory')
        
        # Additional data plots (bottom row)
        self.ax_data1 = self.fig.add_subplot(gs[1, 0])
        self.ax_data1.set_title('FSR (Tactile) Data')
        
        self.ax_data2 = self.fig.add_subplot(gs[1, 1])  
        self.ax_data2.set_title('Hand Joint Angles (First 6)')
        
        # Control panel (bottom)
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
    def setup_widgets(self):
        """Setup interactive widgets."""
        # Time slider
        slider_ax = plt.axes([0.1, 0.05, 0.6, 0.03])
        self.time_slider = Slider(
            slider_ax, 'Frame', 0, self.total_frames - 1, 
            valinit=0, valfmt='%d', valstep=1
        )
        self.time_slider.on_changed(self.update_frame)
        
        # Play/Pause button
        play_ax = plt.axes([0.75, 0.05, 0.08, 0.04])
        self.play_button = Button(play_ax, 'Play')
        self.play_button.on_clicked(self.toggle_play)
        
        # Reset button
        reset_ax = plt.axes([0.85, 0.05, 0.08, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_view)
        
    def update_frame(self, frame_idx=None):
        """Update display for given frame."""
        if frame_idx is None:
            frame_idx = int(self.time_slider.val)
        else:
            frame_idx = int(frame_idx)
            
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        
        # Update RGB image (convert BGR to RGB if needed)
        self.ax_rgb.clear()
        rgb_img = self.rgb_data[self.current_frame]
        # Convert BGR to RGB for proper color display
        rgb_img_corrected = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        self.ax_rgb.imshow(rgb_img_corrected)
        self.ax_rgb.set_title(f'RGB Frame {self.current_frame}')
        self.ax_rgb.axis('off')
        
        # Update 3D pose trajectory
        self.update_3d_pose()
        
        # Update additional data plots
        self.update_data_plots()
        
        # Only update slider if value is different to avoid recursion
        if frame_idx is not None and int(self.time_slider.val) != self.current_frame:
            self.time_slider.set_val(self.current_frame)
        
        self.fig.canvas.draw_idle()
        
    def update_3d_pose(self):
        """Update 3D pose visualization."""
        self.ax_3d.clear()
        
        # Plot trajectory up to current frame
        if self.current_frame > 0:
            trajectory = self.positions[:self.current_frame + 1]
            self.ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                           'b-', alpha=0.6, linewidth=2, label='Trajectory')
            
        # Plot current position as large point
        current_pos = self.positions[self.current_frame]
        self.ax_3d.scatter(current_pos[0], current_pos[1], current_pos[2], 
                          c='red', s=100, label='Current Position')
        
        # Plot orientation as arrow (simplified)
        current_orient = self.orientations[self.current_frame]
        arrow_length = 0.05
        self.ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2],
                         arrow_length * np.cos(current_orient[0]),
                         arrow_length * np.sin(current_orient[1]), 
                         arrow_length * current_orient[2],
                         color='green', arrow_length_ratio=0.3)
        
        # Set axis labels and limits
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y') 
        self.ax_3d.set_zlabel('Z')
        
        # Auto-scale axes based on trajectory
        margin = 0.1
        x_range = [self.positions[:, 0].min() - margin, self.positions[:, 0].max() + margin]
        y_range = [self.positions[:, 1].min() - margin, self.positions[:, 1].max() + margin]  
        z_range = [self.positions[:, 2].min() - margin, self.positions[:, 2].max() + margin]
        
        self.ax_3d.set_xlim(x_range)
        self.ax_3d.set_ylim(y_range)
        self.ax_3d.set_zlim(z_range)
        
        self.ax_3d.legend()
        self.ax_3d.set_title(f'3D Pose - Frame {self.current_frame}')
        
    def update_data_plots(self):
        """Update additional data visualization."""
        # FSR (Tactile) data only
        self.ax_data1.clear()
        
        # Plot FSR data with different colors
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i in range(self.fsr_data.shape[1]):
            color = colors[i % len(colors)]
            self.ax_data1.plot(self.fsr_data[:, i], 
                              label=f'FSR Sensor {i}', 
                              color=color,
                              linewidth=2,
                              alpha=0.8)
        
        # Highlight current frame
        self.ax_data1.axvline(x=self.current_frame, color='red', linestyle='-', alpha=0.8, linewidth=2)
        self.ax_data1.set_xlabel('Frame')
        self.ax_data1.set_ylabel('FSR Value')
        # Place legend inside the plot area to avoid overlap
        self.ax_data1.legend(loc='upper right', fontsize=9)
        self.ax_data1.grid(True, alpha=0.3)
        self.ax_data1.set_title('FSR (Tactile) Data')
        
        # Hand joint angles - only first 6
        self.ax_data2.clear()
        joint_names = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5']
        colors2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Only plot first 6 joint angles
        n_joints = min(6, self.hand_action.shape[1])
        for i in range(n_joints):
            self.ax_data2.plot(self.hand_action[:, i], 
                              label=joint_names[i],
                              color=colors2[i],
                              linewidth=2,
                              alpha=0.8)
            
        self.ax_data2.axvline(x=self.current_frame, color='red', linestyle='-', alpha=0.8, linewidth=2)
        self.ax_data2.set_xlabel('Frame')
        self.ax_data2.set_ylabel('Joint Angle')
        # Place legend inside the plot area to avoid overlap
        self.ax_data2.legend(loc='upper left', fontsize=9)
        self.ax_data2.grid(True, alpha=0.3)
        self.ax_data2.set_title('Hand Joint Angles (First 6)')
        
    def toggle_play(self, event=None):
        """Toggle play/pause animation."""
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()
            
    def start_animation(self):
        """Start automatic playback."""
        self.is_playing = True
        self.play_button.label.set_text('Pause')
        
        def animate(frame):
            if self.is_playing and self.current_frame < self.total_frames - 1:
                self.update_frame(self.current_frame + 1)
                return []
            else:
                self.stop_animation()
                return []
                
        self.animation = animation.FuncAnimation(
            self.fig, animate, interval=100, blit=False, repeat=False
        )
        
    def stop_animation(self):
        """Stop automatic playback."""
        self.is_playing = False
        self.play_button.label.set_text('Play')
        if self.animation:
            self.animation.event_source.stop()
            
    def reset_view(self, event=None):
        """Reset to first frame."""
        self.stop_animation()
        self.update_frame(0)
        
    def show(self):
        """Display the interactive visualizer."""
        # Initialize with first frame
        self.update_frame(0)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive Zarr Dataset Visualizer')
    parser.add_argument('--zarr_path', '-z', type=str, 
                       default='data/xhand_dataset_aligned.zarr',
                       help='Path to zarr dataset')
    parser.add_argument('--episode', '-e', type=int, default=0,
                       help='Episode ID to visualize')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found at {args.zarr_path}")
        return
        
    try:
        visualizer = InteractiveZarrVisualizer(args.zarr_path, args.episode)
        visualizer.show()
        
    except Exception as e:
        print(f"Error creating visualizer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()