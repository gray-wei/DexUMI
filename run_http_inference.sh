#!/bin/bash
# Script to run HTTP-based inference with DexUMI

# Check if model path is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [checkpoint] [additional_args]"
    echo "Example: $0 /path/to/model 600"
    exit 1
fi

MODEL_PATH=$1
CHECKPOINT=${2:-600}
shift 2

# Activate conda environment
echo "Activating conda environment..."
source ~/mambaforge/etc/profile.d/conda.sh
conda activate dexumi

# Check if franka_server is running
echo "Checking if franka_server is accessible..."
if ! curl -s -X POST http://127.0.0.1:5000/get_handdof > /dev/null 2>&1; then
    echo "Warning: Cannot connect to franka_server at http://127.0.0.1:5000"
    echo "Please make sure to run the following in another terminal:"
    echo "  cd /home/ubuntu/hly/gray_dex_dp/serl_robot_infra/robot_servers"
    echo "  ./launch_right_server.sh"
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Run the HTTP inference script
echo "Starting HTTP-based inference..."
python real_script/eval_policy/eval_with_http.py \
    --model_path "$MODEL_PATH" \
    --ckpt "$CHECKPOINT" \
    --frequency 10 \
    --max_pos_speed 0.1 \
    --max_rot_speed 0.6 \
    --camera_latency 0.185 \
    --hand_action_latency 0.3 \
    --robot_action_latency 0.170 \
    --exec_horizon 8 \
    --http_url "http://127.0.0.1:5000" \
    --http_timeout 1.0 \
    "$@"