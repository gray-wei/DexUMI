#!/bin/bash

# Evaluation script for XHand with Franka using HTTP control

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexumi

# Path to your trained model
MODEL_PATH="/home/ubuntu/hgw/IL/DexUMI/data/weight/vision_tactile_propio"  # TODO: Update this path
CHECKPOINT=600

# Control parameters
FREQUENCY=15  # Control frequency in Hz
EXEC_HORIZON=8  # Number of action steps to execute before re-predicting

# Visualization settings
ENABLE_VISUALIZATION=false  # Set to true to enable real-time camera visualization

# Camera configuration
CAMERA_TYPE="realsense"  # Options: "realsense" or "oak"

# Latency parameters (in seconds)
CAMERA_LATENCY=0.185
HAND_ACTION_LATENCY=0.3
ROBOT_ACTION_LATENCY=0.170

# Video recording path
VIDEO_RECORD_PATH="video_record"

echo "========================================="
echo "DexUMI Evaluation with XHand + Franka"
echo "========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Camera Type: $CAMERA_TYPE"
echo "Frequency: $FREQUENCY Hz"
echo "Execution Horizon: $EXEC_HORIZON steps"
echo ""
echo "Latency Settings:"
echo "  Camera: ${CAMERA_LATENCY}s"
echo "  Hand Action: ${HAND_ACTION_LATENCY}s"
echo "  Robot Action: ${ROBOT_ACTION_LATENCY}s"
echo ""
echo "Key Features:"
echo "✓ Direct Franka ee_pose (no T_ET transformation)"
echo "✓ Fixed initial positions"
echo "✓ HTTP control interface"
echo "✓ RealSense/OAK camera support"
echo "✓ Multi-step action execution"
echo ""
echo "Make sure the robot server is running:"
echo "  python franka_server.py"
echo ""
echo "Press Ctrl+C to abort, or wait 3 seconds to continue..."
echo "========================================="
echo ""

# Wait for user to check
sleep 3

# Run the evaluation script
python real_script/eval_policy/eval_xhand_franka.py \
    --model_path "$MODEL_PATH" \
    --ckpt $CHECKPOINT \
    --frequency $FREQUENCY \
    --exec_horizon $EXEC_HORIZON \
    --camera_latency $CAMERA_LATENCY \
    --hand_action_latency $HAND_ACTION_LATENCY \
    --robot_action_latency $ROBOT_ACTION_LATENCY