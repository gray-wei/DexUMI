"""
Convert pickle format data to zarr format for DexUMI training

This script converts collected data from pickle format to zarr format that is compatible
with DexUMI training pipeline.

Input structure (pickle):
    collected_data/
    └── episode_0/
        ├── camera_0/
        │   ├── rgb.pkl               # [T, H, W, 3]
        │   └── receive_time.pkl      # [T]
        ├── pose.pkl                  # [T, 7] (xyz + quaternion[xyzw])
        ├── hand_action.pkl           # [T, 12]
        ├── proprioception.pkl        # [T, 14]
        ├── fsr.pkl                   # [T, 5, 3]
        └── timestamps.pkl            # dict with various timestamps

Output structure (zarr):
    dataset.zarr/
    └── episode_0/
        ├── pose                      # [T, 6] (xyz + rotation vector)
        ├── hand_action               # [T, 12]
        ├── proprioception           # [T, 14]
        ├── fsr                      # [T, 3] (averaged across fingers)
        ├── camera_0/
        │   └── rgb                  # [T, H, W, 3]
        └── camera_1/                # (if available)
            └── rgb                  # [T, H, W, 3]
"""

import os
import pickle
import numpy as np
import zarr
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def load_timestamps(timestamp_path: Path) -> np.ndarray:
    """
    Load timestamps from pickle file
    
    Args:
        timestamp_path: Path to timestamp pickle file
    
    Returns:
        timestamps: Array of timestamps in seconds
    """
    with open(timestamp_path, "rb") as f:
        timestamps = pickle.load(f)
    
    # Handle different timestamp formats
    if isinstance(timestamps, dict):
        # For main timestamps.pkl files
        if 'main_timestamps' in timestamps:
            return timestamps['main_timestamps']
        elif 'robot_state_timestamps' in timestamps:
            return timestamps['robot_state_timestamps']
        else:
            # Return first available timestamp array
            for value in timestamps.values():
                if isinstance(value, np.ndarray):
                    return value
    elif isinstance(timestamps, np.ndarray):
        # For direct timestamp arrays (like receive_time.pkl)
        # Convert from milliseconds to seconds if needed
        if timestamps.max() > 1e10:  # Likely milliseconds
            return timestamps / 1000.0
        return timestamps
    
    raise ValueError(f"Unsupported timestamp format in {timestamp_path}")


def find_nearest_timestamps(target_timestamps: np.ndarray, source_timestamps: np.ndarray) -> np.ndarray:
    """
    Find nearest timestamp indices using binary search for efficiency
    
    Args:
        target_timestamps: Target timestamps to match to (e.g., robot timestamps)
        source_timestamps: Source timestamps to match from (e.g., camera timestamps)
    
    Returns:
        matched_indices: Indices in source_timestamps that best match target_timestamps
    """
    matched_indices = np.zeros(len(target_timestamps), dtype=int)
    
    for i, target_ts in enumerate(target_timestamps):
        # Find closest timestamp using binary search
        idx = np.searchsorted(source_timestamps, target_ts)
        
        # Handle boundary cases
        if idx == 0:
            matched_indices[i] = 0
        elif idx == len(source_timestamps):
            matched_indices[i] = len(source_timestamps) - 1
        else:
            # Choose the closer one
            if abs(source_timestamps[idx-1] - target_ts) <= abs(source_timestamps[idx] - target_ts):
                matched_indices[i] = idx - 1
            else:
                matched_indices[i] = idx
    
    return matched_indices


def align_multimodal_episode(episode_path: Path, camera_ids: Optional[List[int]] = None) -> Dict:
    """
    Align multimodal data for a single episode using timestamp matching
    
    Args:
        episode_path: Path to episode directory
        camera_ids: List of camera IDs to include (None = all cameras)
    
    Returns:
        Dict containing aligned episode data
    """
    # Load robot timestamps (low frequency reference)
    robot_timestamps = load_timestamps(episode_path / "timestamps.pkl")
    
    # Load robot state data
    with open(episode_path / "pose.pkl", "rb") as f:
        pose_data = pickle.load(f)
        positions = pose_data[:, :3]
        quaternions = pose_data[:, 3:7]
        rotvec_angles = np.array([quaternion_to_rotvec(q) for q in quaternions])
        aligned_pose = np.concatenate([positions, rotvec_angles], axis=-1).astype(np.float32)
    
    with open(episode_path / "hand_action.pkl", "rb") as f:
        aligned_hand_action = pickle.load(f).astype(np.float32)
    
    with open(episode_path / "proprioception.pkl", "rb") as f:
        aligned_proprioception = pickle.load(f).astype(np.float32)
    
    with open(episode_path / "fsr.pkl", "rb") as f:
        fsr_data = pickle.load(f)
        aligned_fsr = np.mean(fsr_data, axis=1).astype(np.float32)
    
    # Verify robot data consistency
    robot_length = len(aligned_pose)
    for data, name in [(aligned_hand_action, "hand_action"), (aligned_proprioception, "proprioception"), (aligned_fsr, "fsr")]:
        if len(data) != robot_length:
            print(f"Warning: {name} length ({len(data)}) doesn't match pose length ({robot_length})")
    
    # Align camera data
    aligned_cameras = {}
    for cam_dir in episode_path.glob("camera_*"):
        if cam_dir.is_dir():
            cam_id = int(cam_dir.name.split("_")[1])
            
            # Filter by camera_ids if specified
            if camera_ids is not None and cam_id not in camera_ids:
                print(f"  Skipping camera {cam_id} (not in selected camera IDs: {camera_ids})")
                continue
            
            # Load camera timestamps and data
            camera_timestamps = load_timestamps(cam_dir / "receive_time.pkl")
            
            with open(cam_dir / "rgb.pkl", "rb") as f:
                rgb_data = pickle.load(f)
                if rgb_data.dtype != np.uint8:
                    rgb_data = (rgb_data * 255).astype(np.uint8)
            
            # Find matching camera frames for each robot timestamp
            matched_indices = find_nearest_timestamps(robot_timestamps, camera_timestamps)
            
            # Align camera data to robot frequency
            aligned_rgb = rgb_data[matched_indices]
            aligned_cameras[f"camera_{cam_id}"] = aligned_rgb
            
            # Print alignment info
            max_time_diff = np.max(np.abs(camera_timestamps[matched_indices] - robot_timestamps))
            print(f"  Camera {cam_id}: {len(rgb_data)} -> {len(aligned_rgb)} frames, max time diff: {max_time_diff:.3f}s")
    
    return {
        "pose": aligned_pose,
        "hand_action": aligned_hand_action,
        "proprioception": aligned_proprioception,
        "fsr": aligned_fsr,
        "cameras": aligned_cameras
    }


def quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to rotation vector (axis-angle representation)

    Args:
        quat: Array of shape (..., 4) with quaternion in (x, y, z, w) format (as from robot API)

    Returns:
        rotvec: Array of shape (..., 3) with rotation vector in radians
    """
    # Ensure input is numpy array
    quat = np.asarray(quat)

    # Handle both single quaternion and batch
    original_shape = quat.shape
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)

    # Input is already in (x, y, z, w) format for scipy
    quat_scipy = quat

    # Create rotation object and get rotation vector
    r = Rotation.from_quat(quat_scipy)
    rotvec = r.as_rotvec()

    # Restore original shape
    if len(original_shape) == 1:
        rotvec = rotvec.squeeze(0)

    return rotvec


def load_pickle_episode(episode_path: Path, multimodal_format: bool = False, camera_ids: Optional[List[int]] = None) -> Dict:
    """
    Load a single episode from pickle files
    
    Args:
        episode_path: Path to episode directory containing pickle files
        multimodal_format: Whether to use XhandData_Multimodal format with timestamp alignment
        camera_ids: List of camera IDs to include (None = all cameras)
    
    Returns:
        Dict containing all episode data
    """
    if multimodal_format:
        return align_multimodal_episode(episode_path, camera_ids)
    
    # Original format loading
    data = {}
    
    # Load core data files
    with open(episode_path / "pose.pkl", "rb") as f:
        pose_data = pickle.load(f)
        # Convert quaternion (xyz + quat_xyzw) to 6DoF (xyz + rotvec_xyz)
        positions = pose_data[:, :3]  # xyz positions
        quaternions = pose_data[:, 3:7]  # quaternion (x, y, z, w)
        rotvec_angles = np.array([quaternion_to_rotvec(q) for q in quaternions])
        data["pose"] = np.concatenate([positions, rotvec_angles], axis=-1).astype(np.float32)
    
    with open(episode_path / "hand_action.pkl", "rb") as f:
        data["hand_action"] = pickle.load(f).astype(np.float32)
    
    with open(episode_path / "proprioception.pkl", "rb") as f:
        data["proprioception"] = pickle.load(f).astype(np.float32)
    
    with open(episode_path / "fsr.pkl", "rb") as f:
        fsr_data = pickle.load(f)  # Shape: [T, 5, 3]
        # Average across fingers to get [T, 3]
        data["fsr"] = np.mean(fsr_data, axis=1).astype(np.float32)
    
    # Load camera data
    cameras = {}
    for cam_dir in episode_path.glob("camera_*"):
        if cam_dir.is_dir():
            cam_id = int(cam_dir.name.split("_")[1])
            
            # Filter by camera_ids if specified
            if camera_ids is not None and cam_id not in camera_ids:
                print(f"  Skipping camera {cam_id} (not in selected camera IDs: {camera_ids})")
                continue
            
            rgb_path = cam_dir / "rgb.pkl"
            if rgb_path.exists():
                with open(rgb_path, "rb") as f:
                    rgb_data = pickle.load(f)
                    # Ensure RGB data is uint8 and has correct shape
                    if rgb_data.dtype != np.uint8:
                        rgb_data = (rgb_data * 255).astype(np.uint8)
                    cameras[f"camera_{cam_id}"] = rgb_data
    
    data["cameras"] = cameras
    
    # Verify data consistency
    episode_length = len(data["pose"])
    for key in ["hand_action", "proprioception", "fsr"]:
        if len(data[key]) != episode_length:
            print(f"Warning: {key} length ({len(data[key])}) doesn't match pose length ({episode_length})")
    
    return data


def detect_data_format(input_path: Path) -> bool:
    """
    Detect if the data is in XhandData_Multimodal format
    
    Args:
        input_path: Path to input directory
    
    Returns:
        True if multimodal format, False if original format
    """
    # Check first episode for multimodal format indicators
    episode_dirs = sorted([d for d in input_path.glob("episode_*") if d.is_dir()])
    if not episode_dirs:
        return False
    
    first_episode = episode_dirs[0]
    
    # Check for multimodal format indicators
    has_timestamps = (first_episode / "timestamps.pkl").exists()
    has_camera_timestamps = any((cam_dir / "receive_time.pkl").exists() 
                               for cam_dir in first_episode.glob("camera_*") 
                               if cam_dir.is_dir())
    
    return has_timestamps and has_camera_timestamps


def create_zarr_dataset(
    input_dir: str,
    output_path: str,
    episode_ids: Optional[List[int]] = None,
    compression: str = "blosc",
    overwrite: bool = False,
    multimodal_format: Optional[bool] = None,
    camera_ids: Optional[List[int]] = None
) -> None:
    """
    Convert pickle episodes to zarr format
    
    Args:
        input_dir: Directory containing pickle episodes
        output_path: Path to output zarr file
        episode_ids: List of episode IDs to convert (None = all)
        compression: Compression algorithm for zarr
        overwrite: Whether to overwrite existing zarr file
        multimodal_format: Whether to use XhandData_Multimodal format (None = auto-detect)
        camera_ids: List of camera IDs to include (None = all cameras)
    """
    input_path = Path(input_dir)
    
    # Auto-detect data format if not specified
    if multimodal_format is None:
        multimodal_format = detect_data_format(input_path)
        format_type = "XhandData_Multimodal" if multimodal_format else "Original"
        print(f"Auto-detected data format: {format_type}")
    
    # Find all episodes
    if episode_ids is None:
        episode_dirs = sorted([d for d in input_path.glob("episode_*") if d.is_dir()])
        episode_ids = [int(d.name.split("_")[1]) for d in episode_dirs]
    else:
        episode_dirs = [input_path / f"episode_{i}" for i in episode_ids]
    
    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        return
    
    print(f"Found {len(episode_dirs)} episodes to convert")
    if multimodal_format:
        print("Using multimodal format with timestamp alignment")
    if camera_ids is not None:
        print(f"Filtering cameras: only including camera IDs {camera_ids}")
    
    # Create or open zarr file
    if overwrite and os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=overwrite)
    
    # Process each episode
    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        episode_name = episode_dir.name
        print(f"\nProcessing {episode_name}...")
        
        try:
            # Load episode data
            episode_data = load_pickle_episode(episode_dir, multimodal_format=multimodal_format, camera_ids=camera_ids)
            
            # Create episode group in zarr
            episode_group = root.create_group(episode_name, overwrite=True)
            
            # Save core data
            # Pose: [T, 6] (xyz + rotation vector)
            episode_group.create_dataset(
                "pose",
                data=episode_data["pose"],
                chunks=(100, 6),
                dtype=np.float32,
                compressor=zarr.Blosc(cname=compression, clevel=5, shuffle=1)
            )
            
            # Hand action: [T, 12]
            episode_group.create_dataset(
                "hand_action",
                data=episode_data["hand_action"],
                chunks=(100, 12),
                dtype=np.float32,
                compressor=zarr.Blosc(cname=compression, clevel=5, shuffle=1)
            )
            
            # Proprioception: [T, 14]
            episode_group.create_dataset(
                "proprioception",
                data=episode_data["proprioception"],
                chunks=(100, 14),
                dtype=np.float32,
                compressor=zarr.Blosc(cname=compression, clevel=5, shuffle=1)
            )
            
            # FSR: [T, 3]
            episode_group.create_dataset(
                "fsr",
                data=episode_data["fsr"],
                chunks=(100, 3),
                dtype=np.float32,
                compressor=zarr.Blosc(cname=compression, clevel=5, shuffle=1)
            )
            
            # Save camera data
            for cam_name, rgb_data in episode_data["cameras"].items():
                cam_group = episode_group.create_group(cam_name)
                
                # RGB data: [T, H, W, C]
                cam_group.create_dataset(
                    "rgb",
                    data=rgb_data,
                    chunks=(10, rgb_data.shape[1], rgb_data.shape[2], 3),
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname=compression, clevel=5, shuffle=1)
                )
            
            # Print episode info
            print(f"  - Pose shape: {episode_data['pose'].shape}")
            print(f"  - Hand action shape: {episode_data['hand_action'].shape}")
            print(f"  - Proprioception shape: {episode_data['proprioception'].shape}")
            print(f"  - FSR shape: {episode_data['fsr'].shape}")
            for cam_name, rgb_data in episode_data["cameras"].items():
                print(f"  - {cam_name} RGB shape: {rgb_data.shape}")
            
        except Exception as e:
            print(f"Error processing {episode_name}: {e}")
            continue
    
    print(f"\n✓ Conversion complete! Zarr dataset saved to: {output_path}")
    
    # Print dataset summary
    print("\nDataset summary:")
    total_frames = 0
    for episode_name in root.group_keys():
        episode = root[episode_name]
        frames = len(episode["pose"])
        total_frames += frames
        print(f"  - {episode_name}: {frames} frames")
    print(f"  Total: {len(list(root.group_keys()))} episodes, {total_frames} frames")


def verify_zarr_dataset(zarr_path: str, num_samples: int = 3) -> None:
    """
    Verify the converted zarr dataset
    
    Args:
        zarr_path: Path to zarr dataset
        num_samples: Number of sample frames to check
    """
    print(f"\nVerifying zarr dataset: {zarr_path}")
    
    root = zarr.open(zarr_path, mode='r')
    episodes = list(root.group_keys())
    
    print(f"Found {len(episodes)} episodes")
    
    for episode_name in episodes[:num_samples]:
        print(f"\n{episode_name}:")
        episode = root[episode_name]
        
        # Check all expected keys
        expected_keys = ["pose", "hand_action", "proprioception", "fsr"]
        for key in expected_keys:
            if key in episode:
                data = episode[key]
                print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
                # Print sample values
                if len(data) > 0:
                    print(f"    Sample: {data[0][:5]}...")
            else:
                print(f"  - {key}: MISSING")
        
        # Check cameras
        for key in episode.group_keys():
            if key.startswith("camera_"):
                cam_group = episode[key]
                if "rgb" in cam_group:
                    rgb_data = cam_group["rgb"]
                    print(f"  - {key}/rgb: shape={rgb_data.shape}, dtype={rgb_data.dtype}")
                    # Check value range
                    if len(rgb_data) > 0:
                        print(f"    Value range: [{rgb_data[0].min()}, {rgb_data[0].max()}]")


def main():
    parser = argparse.ArgumentParser(description="Convert pickle data to zarr format for DexUMI")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="collected_data",
        help="Input directory containing pickle episodes"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dataset.zarr",
        help="Output zarr file path"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode IDs to convert (default: all)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="blosclz",
        choices=["blosclz", "zstd", "lz4"],
        help="Compression algorithm"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zarr file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the converted dataset"
    )
    parser.add_argument(
        "--multimodal_format",
        action="store_true",
        help="Force XhandData_Multimodal format with timestamp alignment (default: auto-detect)"
    )
    parser.add_argument(
        "--camera_ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific camera IDs to include (e.g., --camera_ids 0 1). Default: all cameras"
    )
    
    args = parser.parse_args()
    
    # Convert data
    create_zarr_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        episode_ids=args.episodes,
        compression=args.compression,
        overwrite=args.overwrite,
        multimodal_format=args.multimodal_format if args.multimodal_format else None,
        camera_ids=args.camera_ids
    )
    
    # Verify if requested
    if args.verify:
        verify_zarr_dataset(args.output_path)


if __name__ == "__main__":
    main()