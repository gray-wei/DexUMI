#!/bin/bash

# Evaluation script for XHand with Franka using HTTP control

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexumi

# Set Python path for dexumi module
export PYTHONPATH="/home/ubuntu/hgw/IL/DexUMI:$PYTHONPATH"

# Path to your trained model
MODEL_PATH="/home/ubuntu/hgw/IL/DexUMI/data/weight/vision_only_0909"
CHECKPOINT=100

# Control parameters
FREQUENCY=20  # Control frequency in Hz
EXEC_HORIZON=8  # Number of action steps to execute before re-predicting
SESSION_DURATION=120.0  # Session duration in seconds

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
echo "Session Duration: $SESSION_DURATION seconds"
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
echo "Note: Robot server will be checked during runtime"
echo ""
echo "Starting in 3 seconds... (Press Ctrl+C to abort)"
echo "========================================="
echo ""

# Countdown
for i in 3 2 1; do
    echo -n "$i... "
    sleep 1
done
echo "Starting!"

# Run the evaluation script
python real_script/eval_policy/eval_xhand_franka.py \
    --model_path "$MODEL_PATH" \
    --ckpt $CHECKPOINT \
    --frequency $FREQUENCY \
    --exec_horizon $EXEC_HORIZON \
    --session_duration $SESSION_DURATION \
    --camera_latency $CAMERA_LATENCY \
    --hand_action_latency $HAND_ACTION_LATENCY \
    --robot_action_latency $ROBOT_ACTION_LATENCY