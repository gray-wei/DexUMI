# Pickle to Zarr Data Conversion Script

## Overview
This script converts collected data from pickle format to zarr format that is compatible with DexUMI training pipeline.

## Usage

### Basic conversion
```bash
python convert_pickle_to_zarr.py --input_dir collected_data --output_path dataset.zarr
```

### With verification
```bash
python convert_pickle_to_zarr.py --input_dir collected_data --output_path dataset.zarr --verify
```

### Convert specific episodes
```bash
python convert_pickle_to_zarr.py --input_dir collected_data --output_path dataset.zarr --episodes 0 1 2
```

### Overwrite existing dataset
```bash
python convert_pickle_to_zarr.py --input_dir collected_data --output_path dataset.zarr --overwrite
```

### Use different compression
```bash
python convert_pickle_to_zarr.py --input_dir collected_data --output_path dataset.zarr --compression zstd
```

## Arguments

- `--input_dir`: Directory containing pickle episodes (default: `collected_data`)
- `--output_path`: Output zarr file path (default: `dataset.zarr`)
- `--episodes`: Specific episode IDs to convert (optional, default: all)
- `--compression`: Compression algorithm (choices: blosclz, zstd, lz4, default: blosclz)
- `--overwrite`: Overwrite existing zarr file
- `--verify`: Verify the converted dataset after conversion

## Data Format Conversion

### Input (Pickle)
```
collected_data/
└── episode_0/
    ├── camera_0/
    │   ├── rgb.pkl               # [T, H, W, 3]
    │   └── receive_time.pkl      # [T]
    ├── pose.pkl                  # [T, 7] (xyz + quaternion wxyz)
    ├── hand_action.pkl           # [T, 12]
    ├── proprioception.pkl        # [T, 14]
    ├── fsr.pkl                   # [T, 5, 3]
    └── timestamps.pkl            # dict with various timestamps
```

### Output (Zarr)
```
dataset.zarr/
└── episode_0/
    ├── pose                      # [T, 6] (xyz + euler angles xyz)
    ├── hand_action               # [T, 12]
    ├── proprioception           # [T, 14]
    ├── fsr                      # [T, 3] (averaged across fingers)
    └── camera_0/
        └── rgb                  # [T, H, W, 3]
```

## Key Transformations

1. **Pose**: Quaternion (w,x,y,z) → Euler angles (roll, pitch, yaw)
2. **FSR**: Averaged across 5 fingers: [T, 5, 3] → [T, 3]
3. **Camera**: RGB images resized to 256x256 if needed
4. **Data types**: 
   - Pose, hand_action, proprioception, fsr: float32
   - RGB images: uint8

## Integration with DexUMI

The converted zarr dataset can be directly used with DexUMI training:

```python
from dexumi.diffusion_policy.dataloader.replay_buffer import DexUMIReplayBuffer

# Load the converted dataset
replay_buffer = DexUMIReplayBuffer(
    data_path=["dataset.zarr"],
    load_camera_ids=[0],  # Load camera_0
    camera_resize_shape=[256, 256],
    enable_fsr=True,
    fsr_binary_cutoff=[0.5, 0.5, 0.5]  # Optional FSR thresholding
)
```

## Troubleshooting

1. **Missing data**: Check that all pickle files exist in the episode directories
2. **Dimension mismatch**: Verify that all frames in an episode have consistent dimensions
3. **Memory issues**: For large datasets, process episodes in batches
4. **Compression errors**: Make sure to use supported compressors (blosclz, zstd, lz4)