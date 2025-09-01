# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@/home/gray/.claude/CLAUDE.md

## Project Overview

DexUMI (Dexterous Manipulation Universal Interface) is a comprehensive system for transferring human manipulation skills to robotic dexterous hands through exoskeleton-based demonstration collection, data processing, and diffusion policy training. The system supports both XHand and Inspire Hand robotic platforms.

## Core Architecture

### Multi-Stage Data Pipeline
The system follows a sequential data processing pipeline from human demonstration to robot deployment:

1. **Exoskeleton Data Collection** → Raw human demonstrations with force/tactile feedback
2. **Data Interpolation & Replay** → Convert exoskeleton motions to robot hand motions  
3. **Vision Processing** → SAM2 segmentation + ProPainter inpainting to remove exoskeleton
4. **Dataset Generation** → Structured datasets for policy training
5. **Diffusion Policy Training** → Vision-based imitation learning
6. **Robot Deployment** → Real-time policy execution

### Key Abstractions

**DexterousHand Interface** (`dexumi/hand_sdk/dexhand.py`): Abstract base class defining the standard interface for all robotic hands. Concrete implementations:
- `InspireSDK` - Inspire Hand control via serial communication
- `XHandSDK` - XHand control with specialized tactile feedback

**Diffusion Policy Architecture** (`dexumi/diffusion_policy/`): Vision-transformer backbone + conditional U-Net for action generation:
- Vision encoder: DINO ViT-Small processes visual observations
- Global conditioning: Combines visual features, proprioception, and force sensor data
- Action prediction: 16-step horizon with 8-step action chunks

**Multi-Modal Sensor Integration**: Combines visual observations, proprioceptive feedback, and optional force sensor readings (FSR) for policy conditioning.

## Development Commands

### Environment Setup
```bash
# Main development environment
mamba env create -f environment.yml
mamba activate dexumi

# Hardware optimization environment (separate to avoid conflicts)
mamba env create -f environment_design.yml
mamba activate dexumi_design
```

### Data Collection & Processing
```bash
# Record exoskeleton demonstrations
python real_script/data_collection/record_exoskeleton.py -et -ef --fps 45 \
  --reference-dir /path/to/reference --hand_type xhand/inspire --data-dir /path/to/data

# Process collected data through full pipeline
cd real_script/data_generation_pipeline
# Update paths in process.sh: DATA_DIR, TARGET_DIR, REFERENCE_DIR
./process.sh

# Generate final training dataset
python 6_generate_dataset.py -d /path/to/data_replay -t /path/to/final_dataset \
  --force-process total --force-adjust
```

### Policy Training
```bash
# Configure training parameters in config/diffusion_policy/train_diffusion_policy.yaml:
# - dataset.data_dirs: ["/path/to/final_dataset"]
# - dataset.enable_fsr: true/false
# - model.global_cond_dim: 384 + force_sensor_dims

# Launch distributed training
accelerate launch real_script/policy_training/train_diffusion_policy.py
```

### Robot Deployment
```bash
# Start robot control server
python real_script/open_server.py --dexhand --ur5

# Evaluate trained policy
python real_script/eval_policy/eval_xhand.py --model_path /path/to/model --ckpt N
python real_script/eval_policy/eval_inspire.py --model_path /path/to/model --ckpt N
```

### Hardware Optimization
```bash
# Switch to design environment first
mamba activate dexumi_design

# Visualize motion capture trajectories
python linkage_optimization/viz_multi_fingertips_trajectory.py

# Generate linkage design candidates
python linkage_optimization/sweep_valid_linkage_design.py --type finger/thumb --save_path /path/to/sim

# Optimize linkage parameters
python linkage_optimization/get_equivalent_finger.py -r /path/to/sim -b /path/to/mocap
python linkage_optimization/get_equivalent_thumb.py -r /path/to/sim -b /path/to/mocap
```

## Configuration Architecture

**Hydra Configuration System**: Uses hierarchical YAML configs with environment variable substitution:
- `config/diffusion_policy/train_diffusion_policy.yaml` - Main training configuration
- `config/render/render_all_dataset.yaml` - Dataset generation parameters

**Hand-Specific Constants** (`dexumi/constants.py`): Motor scaling factors, control parameters, and hand-specific adjustments for both XHand and Inspire Hand platforms.

**Force Sensor Integration**: Optional FSR (Force Sensitive Resistor) support with configurable binary cutoff thresholds. XHand uses [10,10,10] threshold; Inspire Hand requires installation-specific calibration.

## System Integration Points

**External Dependencies**: 
- SAM2 (segment-anything-2) for exoskeleton segmentation
- ProPainter for video inpainting  
- Record3D for iPhone-based wrist pose tracking
- Custom spnav fork for 3D mouse support

**Embedded System**: STM32-based force sensor board with custom firmware (`embedded_system/`) supporting both XHand and Inspire Hand configurations.

**Robot Integration**: UR5 robot arm integration with hand-eye calibration system for coordinated manipulation tasks.

## Key Development Patterns

**Hand Type Abstraction**: All hand-specific logic is encapsulated in SDK classes. When adding new hand types, implement the `DexterousHand` interface and update constants in `dexumi/constants.py`.

**Multi-Modal Data Handling**: Vision, proprioception, and force data are processed separately then concatenated for policy conditioning. Missing modalities are handled gracefully.

**Distributed Training**: Uses Accelerate for multi-GPU training with NCCL backend. Training supports gradient accumulation and EMA model updates.

**Real-Time Control**: Policy evaluation runs in real-time with 30Hz control loop, requiring careful attention to inference latency and action smoothing.