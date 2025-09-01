#!/bin/bash

# FIXED evaluation script for RealSense HTTP control

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dexumi

# Path to your trained model
MODEL_PATH="/home/ubuntu/hgw/IL/DexUMI/data/weight/vision_tactile_propio"
CHECKPOINT=600

# Control parameters
FREQUENCY=10  # Control frequency in Hz
EXEC_HORIZON=8  # Number of action steps to execute before re-predicting

echo "========================================="
echo "DexUMI Evaluation with RealSense (FIXED)"
echo "========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Frequency: $FREQUENCY Hz"
echo "Execution Horizon: $EXEC_HORIZON steps"
echo ""
echo "Critical fixes applied:"
echo "✓ T_ET coordinate transformation"
echo "✓ BGR color format (no RGB conversion)"
echo "✓ Multi-step action execution"
echo "✓ Proper state tracking"
echo ""
echo "Make sure the robot server is running:"
echo "  ./launch_right_server.sh"
echo ""
echo "========================================="
echo ""

# Run the fixed evaluation script
python eval_realsense_http_fixed.py \
    -mp "$MODEL_PATH" \
    -ckpt $CHECKPOINT \
    -f $FREQUENCY \
    -eh $EXEC_HORIZON \
    --debug