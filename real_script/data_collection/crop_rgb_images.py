#!/usr/bin/env python3
"""
Batch crop RGB images from camera_0/rgb.pkl files
Crops images to 160x240 pixels from top-left corner (0,0)
"""

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def crop_image(image, width=160, height=240):
    """Crop image from top-left corner to specified dimensions"""
    if len(image.shape) == 3:  # RGB image
        return image[:height, :width, :]
    elif len(image.shape) == 2:  # Grayscale
        return image[:height, :width]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

def process_rgb_file(rgb_input_path, rgb_output_path):
    """Process a single rgb.pkl file and save cropped version"""
    try:
        # Load the original pickle file
        with open(rgb_input_path, 'rb') as f:
            rgb_data = pickle.load(f)

        # Handle different data structures
        if isinstance(rgb_data, list):
            # If it's a list of images
            cropped_data = []
            for img in rgb_data:
                cropped_img = crop_image(img)
                cropped_data.append(cropped_img)
        elif isinstance(rgb_data, np.ndarray):
            # If it's a single image or array of images
            if len(rgb_data.shape) == 4:  # Multiple images (N, H, W, C)
                cropped_data = rgb_data[:, :240, :160, :]
            elif len(rgb_data.shape) == 3:  # Single image (H, W, C)
                cropped_data = crop_image(rgb_data)
            else:
                raise ValueError(f"Unexpected array shape: {rgb_data.shape}")
        else:
            # Try to crop whatever structure it is
            cropped_data = crop_image(rgb_data)

        # Save the cropped data
        with open(rgb_output_path, 'wb') as f:
            pickle.dump(cropped_data, f)

        return True

    except Exception as e:
        print(f"Error processing {rgb_input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch crop RGB images from camera_0/rgb.pkl files')
    parser.add_argument('--base_dir', type=str,
                        default="/home/gray/Project/DexUMI/grasp_veg",
                        help='Base directory containing the data')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite original rgb.pkl files (default: save as rgb_cropped.pkl)')
    parser.add_argument('--width', type=int, default=160,
                        help='Crop width (default: 160)')
    parser.add_argument('--height', type=int, default=240,
                        help='Crop height (default: 240)')
    parser.add_argument('--camera', type=str, default='camera_0',
                        help='Camera directory name (default: camera_0)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return

    # Find all rgb.pkl files in specified camera directories
    rgb_files = list(base_dir.glob(f"*/{args.camera}/rgb.pkl"))

    if len(rgb_files) == 0:
        print(f"No rgb.pkl files found in {base_dir}/*/{args.camera}/")
        return

    print(f"Found {len(rgb_files)} rgb.pkl files to process")
    print(f"Crop dimensions: {args.width}x{args.height}")

    if args.overwrite:
        print("WARNING: Will overwrite original rgb.pkl files")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted")
            return
        output_suffix = ""
    else:
        print("Will save as rgb_cropped.pkl (original files preserved)")
        output_suffix = "_cropped"

    successful = 0
    failed = 0

    # Process each file with progress bar
    for rgb_input_path in tqdm(rgb_files, desc="Processing files"):
        # Generate the output path
        if args.overwrite:
            rgb_output_path = rgb_input_path
        else:
            rgb_output_path = rgb_input_path.parent / "rgb_cropped.pkl"

        if process_rgb_file(rgb_input_path, rgb_output_path):
            successful += 1
        else:
            failed += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total files: {len(rgb_files)}")

if __name__ == "__main__":
    main()