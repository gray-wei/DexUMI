#!/bin/bash

# Advanced evaluation script for XHand with Franka using HTTP control
# Supports multiple configurations and debug modes

set -e  # Exit on error

# ============================================
# Configuration Section
# ============================================

# Model configuration
MODEL_BASE_DIR="/home/gray/Project/DexUMI/data/weight"
MODEL_NAME="vision_tactile_propio"  # Update this to your model name
MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"
CHECKPOINT=600

# Control parameters
FREQUENCY=10  # Control frequency in Hz
EXEC_HORIZON=8  # Number of action steps to execute before re-predicting

# Camera configuration
CAMERA_TYPE="realsense"  # Options: "realsense" or "oak"
ENABLE_RECORD_CAMERA=false  # Set to true if you have a second camera

# Latency parameters (in seconds)
CAMERA_LATENCY=0.185
HAND_ACTION_LATENCY=0.3
ROBOT_ACTION_LATENCY=0.170

# Recording configuration
VIDEO_RECORD_PATH="video_record/$(date +%Y%m%d_%H%M%S)"
MATCH_EPISODE_PATH=""  # Optional: path to reference episodes

# Server configuration
ROBOT_SERVER_URL="http://127.0.0.1:5000"

# Debug mode
DEBUG_MODE=false

# ============================================
# Functions
# ============================================

print_banner() {
    echo "========================================="
    echo "DexUMI Evaluation with XHand + Franka"
    echo "========================================="
}

print_config() {
    echo ""
    echo "Configuration:"
    echo "  Model: $MODEL_PATH"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Camera Type: $CAMERA_TYPE"
    echo "  Record Camera: $ENABLE_RECORD_CAMERA"
    echo "  Frequency: $FREQUENCY Hz"
    echo "  Execution Horizon: $EXEC_HORIZON steps"
    echo ""
    echo "Latency Settings:"
    echo "  Camera: ${CAMERA_LATENCY}s"
    echo "  Hand Action: ${HAND_ACTION_LATENCY}s"
    echo "  Robot Action: ${ROBOT_ACTION_LATENCY}s"
    echo ""
    echo "Recording:"
    echo "  Video Path: $VIDEO_RECORD_PATH"
    if [ -n "$MATCH_EPISODE_PATH" ]; then
        echo "  Match Episode: $MATCH_EPISODE_PATH"
    fi
    echo ""
}

check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ Error: Model path does not exist: $MODEL_PATH"
        exit 1
    fi
    
    # Check if checkpoint exists
    CKPT_FILE="${MODEL_PATH}/ckpt_${CHECKPOINT}.pt"
    if [ ! -f "$CKPT_FILE" ]; then
        echo "⚠️  Warning: Checkpoint file not found: $CKPT_FILE"
        echo "   Available checkpoints:"
        ls -la ${MODEL_PATH}/ckpt_*.pt 2>/dev/null || echo "   No checkpoints found"
    fi
    
    # Check if robot server is running
    echo -n "Checking robot server at $ROBOT_SERVER_URL... "
    if curl -s -o /dev/null -w "%{http_code}" "${ROBOT_SERVER_URL}/health" | grep -q "200"; then
        echo "✅ Connected"
    else
        echo "❌ Not responding"
        echo ""
        echo "Please start the robot server first:"
        echo "  python franka_server.py"
        exit 1
    fi
    
    # Check camera availability
    echo -n "Checking ${CAMERA_TYPE} camera... "
    if [ "$CAMERA_TYPE" == "realsense" ]; then
        if python3 -c "import pyrealsense2" 2>/dev/null; then
            echo "✅ Library available"
        else
            echo "❌ pyrealsense2 not installed"
            exit 1
        fi
    else
        echo "✅ Using OAK camera"
    fi
    
    echo ""
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_NAME     Model name (default: $MODEL_NAME)"
    echo "  -c, --checkpoint CKPT      Checkpoint number (default: $CHECKPOINT)"
    echo "  -f, --frequency HZ         Control frequency (default: $FREQUENCY)"
    echo "  -e, --exec-horizon N       Execution horizon (default: $EXEC_HORIZON)"
    echo "  -t, --camera-type TYPE     Camera type: realsense|oak (default: $CAMERA_TYPE)"
    echo "  -r, --record              Enable record camera"
    echo "  -d, --debug               Enable debug mode"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run with default settings"
    echo "  $0 -m my_model -c 800     # Use specific model and checkpoint"
    echo "  $0 -t oak -r              # Use OAK camera with recording"
    echo "  $0 -d                     # Run in debug mode"
    exit 0
}

# ============================================
# Parse command line arguments
# ============================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -f|--frequency)
            FREQUENCY="$2"
            shift 2
            ;;
        -e|--exec-horizon)
            EXEC_HORIZON="$2"
            shift 2
            ;;
        -t|--camera-type)
            CAMERA_TYPE="$2"
            shift 2
            ;;
        -r|--record)
            ENABLE_RECORD_CAMERA=true
            shift
            ;;
        -d|--debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# ============================================
# Main execution
# ============================================

print_banner

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexumi

print_config

echo "Key Features:"
echo "✓ Direct Franka ee_pose (no T_ET transformation)"
echo "✓ Fixed initial positions"
echo "✓ HTTP control interface"
echo "✓ ${CAMERA_TYPE^} camera support"
echo "✓ Multi-step action execution"
echo "✓ Relative control semantics"
echo ""

check_prerequisites

echo "Starting in 3 seconds... (Press Ctrl+C to abort)"
for i in 3 2 1; do
    echo -n "$i... "
    sleep 1
done
echo ""
echo ""

# Create video recording directory
mkdir -p "$VIDEO_RECORD_PATH"

# Build command
CMD="python real_script/eval_policy/eval_xhand_franka.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --ckpt $CHECKPOINT"
CMD="$CMD --frequency $FREQUENCY"
CMD="$CMD --exec_horizon $EXEC_HORIZON"
CMD="$CMD --camera_type $CAMERA_TYPE"
CMD="$CMD --camera_latency $CAMERA_LATENCY"
CMD="$CMD --hand_action_latency $HAND_ACTION_LATENCY"
CMD="$CMD --robot_action_latency $ROBOT_ACTION_LATENCY"
CMD="$CMD --video_record_path \"$VIDEO_RECORD_PATH\""

if [ "$ENABLE_RECORD_CAMERA" = true ]; then
    CMD="$CMD --enable_record_camera"
fi

if [ -n "$MATCH_EPISODE_PATH" ]; then
    CMD="$CMD --match_episode_path \"$MATCH_EPISODE_PATH\""
fi

# Execute
echo "Executing command:"
echo "$CMD"
echo ""
echo "========================================="
echo ""

if [ "$DEBUG_MODE" = true ]; then
    # Debug mode: run with Python debugger
    python -m pdb real_script/eval_policy/eval_xhand_franka.py \
        --model_path "$MODEL_PATH" \
        --ckpt $CHECKPOINT \
        --frequency $FREQUENCY \
        --exec_horizon $EXEC_HORIZON \
        --camera_type $CAMERA_TYPE \
        --camera_latency $CAMERA_LATENCY \
        --hand_action_latency $HAND_ACTION_LATENCY \
        --robot_action_latency $ROBOT_ACTION_LATENCY \
        --video_record_path "$VIDEO_RECORD_PATH" \
        $([ "$ENABLE_RECORD_CAMERA" = true ] && echo "--enable_record_camera") \
        $([ -n "$MATCH_EPISODE_PATH" ] && echo "--match_episode_path \"$MATCH_EPISODE_PATH\"")
else
    # Normal execution
    eval $CMD
fi

echo ""
echo "========================================="
echo "Evaluation completed!"
echo "Video saved to: $VIDEO_RECORD_PATH"
echo "========================================="