#!/usr/bin/env python3
"""
Export all images from a specific episode in sequential order for inspection.
This script saves every frame from an episode as individual image files.
"""

import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm


def load_pickle_data(file_path):
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def export_episode_sequence(episode_dir, output_dir, image_format='jpg', quality=95):
    """
    Export all images from an episode as individual files.
    
    Args:
        episode_dir: Path to episode directory
        output_dir: Directory to save individual image files
        image_format: Output format ('jpg', 'png')
        quality: JPEG quality (1-100, only for jpg)
    """
    episode_path = Path(episode_dir)
    output_path = Path(output_dir)
    
    # Load RGB images
    rgb_file = episode_path / "camera_0" / "rgb.pkl"
    if not rgb_file.exists():
        print(f"Error: RGB file not found at {rgb_file}")
        return False
    
    # Load timestamps for reference
    timestamps_file = episode_path / "timestamps.pkl"
    timestamps_data = None
    if timestamps_file.exists():
        timestamps_data = load_pickle_data(timestamps_file)
    
    print(f"Loading images from {rgb_file}...")
    images = load_pickle_data(rgb_file)
    
    if images is None or len(images) == 0:
        print("No images found in the episode")
        return False
    
    # Create output directory
    episode_name = episode_path.name
    episode_output_dir = output_path / episode_name
    episode_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {len(images)} images to {episode_output_dir}")
    print(f"Image format: {image_format.upper()}, Quality: {quality if image_format.lower() == 'jpg' else 'N/A'}")
    
    # Export each image
    for i, img in enumerate(tqdm(images, desc="Exporting frames")):
        try:
            # Handle different image formats
            if isinstance(img, np.ndarray):
                # Ensure uint8 format
                if img.dtype != np.uint8:
                    # Normalize to uint8 if needed
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                
                # Convert BGR to RGB for correct color display
                # RealSense camera outputs BGR format, but PIL expects RGB
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[..., ::-1]  # BGR to RGB conversion
                
                # Convert numpy array to PIL Image
                pil_img = Image.fromarray(img)
                
                # Generate filename with frame number (zero-padded)
                filename = f"frame_{i:06d}.{image_format.lower()}"
                filepath = episode_output_dir / filename
                
                # Save image
                if image_format.lower() == 'jpg':
                    pil_img.save(filepath, 'JPEG', quality=quality)
                elif image_format.lower() == 'png':
                    pil_img.save(filepath, 'PNG')
                else:
                    print(f"Unsupported format: {image_format}")
                    return False
                    
            else:
                print(f"Unexpected image format at index {i}: {type(img)}")
                continue
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
    
    # Save metadata file
    metadata_file = episode_output_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Episode: {episode_name}\n")
        f.write(f"Total frames: {len(images)}\n")
        f.write(f"Image format: {image_format.upper()}\n")
        f.write(f"Image shape: {images[0].shape if len(images) > 0 else 'Unknown'}\n")
        f.write(f"Export format: BGR->RGB converted\n")
        
        if timestamps_data and 'main_timestamps' in timestamps_data:
            main_ts = timestamps_data['main_timestamps']
            if len(main_ts) > 1:
                duration = main_ts[-1] - main_ts[0]
                fps = len(main_ts) / duration
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Average FPS: {fps:.2f}\n")
    
    print(f"✓ Successfully exported {len(images)} frames to {episode_output_dir}")
    print(f"✓ Metadata saved to {metadata_file}")
    
    return True


def create_video_from_images(image_dir, output_video_path, fps=20):
    """
    Create a video from exported images using ffmpeg.
    
    Args:
        image_dir: Directory containing sequential images
        output_video_path: Path for output video file
        fps: Frames per second for output video
    """
    try:
        import subprocess
        
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        # Create video from images
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-framerate', str(fps),
            '-i', str(image_dir / 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Video created: {output_video_path}")
            return True
        else:
            print(f"Error creating video: {result.stderr}")
            return False
            
    except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not available. Skipping video creation.")
        print("To create video manually, run:")
        print(f"ffmpeg -framerate {fps} -i {image_dir}/frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export all images from an episode sequence')
    parser.add_argument('--data_dir', type=str, default='../../../collected_data_optimized',
                        help='Path to collected_data_optimized directory')
    parser.add_argument('--episode', type=str, required=True,
                        help='Episode to export (e.g., episode_0)')
    parser.add_argument('--output_dir', type=str, default='./exported_sequences',
                        help='Directory to save exported images')
    parser.add_argument('--format', type=str, choices=['jpg', 'png'], default='jpg',
                        help='Output image format')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (1-100, only for jpg format)')
    parser.add_argument('--create_video', action='store_true',
                        help='Also create a video file from the images')
    parser.add_argument('--fps', type=int, default=20,
                        help='FPS for video creation')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    episode_dir = data_dir / args.episode
    output_dir = Path(args.output_dir)
    
    if not episode_dir.exists():
        print(f"Error: Episode directory {episode_dir} does not exist")
        return
    
    # Export images
    success = export_episode_sequence(
        episode_dir=episode_dir,
        output_dir=output_dir,
        image_format=args.format,
        quality=args.quality
    )
    
    if not success:
        print("Failed to export images")
        return
    
    # Create video if requested
    if args.create_video and args.format.lower() == 'jpg':
        episode_output_dir = output_dir / args.episode
        video_path = episode_output_dir / f"{args.episode}.mp4"
        create_video_from_images(episode_output_dir, video_path, args.fps)
    elif args.create_video:
        print("Video creation only supported for jpg format")
    
    print(f"\n✓ Export complete!")
    print(f"Images saved to: {output_dir / args.episode}")
    if args.create_video and args.format.lower() == 'jpg':
        print(f"Video saved to: {output_dir / args.episode / f'{args.episode}.mp4'}")


if __name__ == '__main__':
    main()