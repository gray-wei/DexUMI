#!/bin/bash

# DexUMI Diffusion Policy Training Script
# Usage: ./train_dexumi.sh

echo "🚀 Starting DexUMI Diffusion Policy Training..."

# Set working directory
cd /home/guowei/Project/DexUMI

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexumi

# Set environment variables for CUDA compatibility
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix for RTX 4000 series GPU communication issues
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set Python path for dexumi module
export PYTHONPATH=/home/guowei/Project/DexUMI:$PYTHONPATH

# Set Hydra configuration paths
export STORE_PATH=/home/guowei/Project/DexUMI
export DEV_PATH=/home/guowei/Project/DexUMI

echo "✓ Environment configured"
echo "✓ Working directory: $(pwd)"
echo "✓ Python path: $PYTHONPATH"
echo "✓ Conda environment: $(conda info --envs | grep '*')"

# Check GPU status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo "🎯 Starting training..."

# Start training with accelerate
accelerate launch real_script/policy_training/train_diffusion_policy.py

# Start training without accelerate
python real_script/policy_training/train_diffusion_policy.py

echo "✅ Training completed or terminated"