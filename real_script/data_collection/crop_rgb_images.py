#!/usr/bin/env python3
"""
Batch crop RGB images from rgb_no_crop.pkl files
Crops images to 193x238 pixels from top-left corner (0,0)
"""

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

def crop_image(image, width=193, height=238):
    """Crop image from top-left corner to specified dimensions"""
    if len(image.shape) == 3:  # RGB image
        return image[:height, :width, :]
    elif len(image.shape) == 2:  # Grayscale
        return image[:height, :width]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

def process_rgb_file(rgb_no_crop_path, rgb_path):
    """Process a single rgb_no_crop.pkl file and save cropped version as rgb.pkl"""
    try:
        # Load the original pickle file
        with open(rgb_no_crop_path, 'rb') as f:
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
                cropped_data = rgb_data[:, :238, :193, :]
            elif len(rgb_data.shape) == 3:  # Single image (H, W, C)
                cropped_data = crop_image(rgb_data)
            else:
                raise ValueError(f"Unexpected array shape: {rgb_data.shape}")
        else:
            # Try to crop whatever structure it is
            cropped_data = crop_image(rgb_data)
        
        # Save the cropped data as rgb.pkl
        with open(rgb_path, 'wb') as f:
            pickle.dump(cropped_data, f)
            
        return True
        
    except Exception as e:
        print(f"Error processing {rgb_no_crop_path}: {e}")
        return False

def main():
    base_dir = Path("/home/gray/dataset/dexumi/XhandData_Multimodal")
    
    # Find all rgb_no_crop.pkl files in camera_1 directories
    rgb_no_crop_files = list(base_dir.glob("*/camera_1/rgb_no_crop.pkl"))
    
    print(f"Found {len(rgb_no_crop_files)} rgb_no_crop.pkl files to process")
    
    successful = 0
    failed = 0
    
    # Process each file with progress bar
    for rgb_no_crop_path in tqdm(rgb_no_crop_files, desc="Processing files"):
        # Generate the corresponding rgb.pkl path
        rgb_path = rgb_no_crop_path.parent / "rgb.pkl"
        
        if process_rgb_file(rgb_no_crop_path, rgb_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total files: {len(rgb_no_crop_files)}")

if __name__ == "__main__":
    main()