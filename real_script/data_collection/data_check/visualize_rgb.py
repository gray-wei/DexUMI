#!/usr/bin/env python3
"""
Script to visualize rgb.pkl files from camera directories and save as videos.
Each episode contains two cameras (camera_0 and camera_1), and this script
processes all episodes in the current directory.

usage:
in episode dir:
python visualize_rgb.py --episode episode_0

if you want to process all episodes, run:
python visualize_rgb.py

"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path
import argparsea

def load_rgb_data(pkl_path):
    """Load RGB data from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

def create_video_from_rgb(rgb_data, output_path, fps=30):
    """Create video from RGB numpy array."""
    if rgb_data is None:
        return False
    
    # Get dimensions
    num_frames, height, width, channels = rgb_data.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {num_frames} frames at {fps} FPS...")
    
    try:
        for i in range(num_frames):
            frame = rgb_data[i]
            # Data is already in correct format, no color conversion needed
            out.write(frame)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == num_frames - 1:
                print(f"  Progress: {i + 1}/{num_frames} frames")
    
    except Exception as e:
        print(f"Error creating video: {e}")
        return False
    finally:
        out.release()
    
    return True

def process_episode(episode_path, fps=30):
    """Process all camera directories in an episode."""
    episode_path = Path(episode_path)
    
    if not episode_path.exists():
        print(f"Episode path does not exist: {episode_path}")
        return
    
    # Find all camera directories
    camera_dirs = [d for d in episode_path.iterdir() 
                   if d.is_dir() and d.name.startswith('camera_')]
    
    if not camera_dirs:
        print(f"No camera directories found in {episode_path}")
        return
    
    print(f"\nProcessing episode: {episode_path.name}")
    print(f"Found {len(camera_dirs)} camera directories")
    
    for camera_dir in sorted(camera_dirs):
        rgb_pkl_path = camera_dir / 'rgb.pkl'
        
        if not rgb_pkl_path.exists():
            print(f"  rgb.pkl not found in {camera_dir}")
            continue
        
        print(f"  Processing {camera_dir.name}...")
        
        # Load RGB data
        rgb_data = load_rgb_data(rgb_pkl_path)
        if rgb_data is None:
            continue
        
        print(f"    Loaded RGB data: {rgb_data.shape} (frames, height, width, channels)")
        
        # Create output video path
        video_filename = f"{camera_dir.name}_rgb_video.mp4"
        output_video_path = camera_dir / video_filename
        
        # Create video
        success = create_video_from_rgb(rgb_data, str(output_video_path), fps)
        
        if success:
            print(f"    ✓ Video saved: {output_video_path}")
        else:
            print(f"    ✗ Failed to create video for {camera_dir}")

def main():
    """Main function to process all episodes in current directory."""
    parser = argparse.ArgumentParser(description='Visualize RGB pickle files as videos')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frames per second for output videos (default: 30)')
    parser.add_argument('--episode', type=str, 
                       help='Process specific episode (e.g., episode_0). If not specified, processes all episodes.')
    
    args = parser.parse_args()
    
    current_dir = Path('.')
    
    if args.episode:
        # Process specific episode
        episode_path = current_dir / args.episode
        if episode_path.exists():
            process_episode(episode_path, args.fps)
        else:
            print(f"Episode directory not found: {args.episode}")
    else:
        # Find all episode directories
        episode_dirs = [d for d in current_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        if not episode_dirs:
            print("No episode directories found in current directory")
            return
        
        print(f"Found {len(episode_dirs)} episode directories")
        
        # Process each episode
        for episode_dir in sorted(episode_dirs):
            process_episode(episode_dir, args.fps)
    
    print("\n🎬 Video visualization complete!")

if __name__ == "__main__":
    main()