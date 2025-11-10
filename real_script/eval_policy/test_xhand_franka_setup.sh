#!/bin/bash

# Quick test script to verify XHand + Franka setup

echo "========================================="
echo "XHand + Franka Setup Test"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test results
print_test() {
    local test_name=$1
    local result=$2
    local message=$3
    
    if [ "$result" = "pass" ]; then
        echo -e "${GREEN}✅ $test_name${NC}"
        ((TESTS_PASSED++))
    elif [ "$result" = "warn" ]; then
        echo -e "${YELLOW}⚠️  $test_name${NC}"
        [ -n "$message" ] && echo "   $message"
    else
        echo -e "${RED}❌ $test_name${NC}"
        [ -n "$message" ] && echo "   $message"
        ((TESTS_FAILED++))
    fi
}

echo "1. Checking Python environment..."
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)" 2>/dev/null; then
    print_test "Python version" "pass"
else
    print_test "Python version" "fail" "Python 3.7+ required"
fi

echo ""
echo "2. Checking required packages..."

# Check DexUMI package
if python3 -c "import dexumi" 2>/dev/null; then
    print_test "DexUMI package" "pass"
else
    print_test "DexUMI package" "fail" "DexUMI not installed or not in PYTHONPATH"
fi

# Check HTTP client
if python3 -c "from dexumi.real_env.common.http_client import HTTPRobotClient" 2>/dev/null; then
    print_test "HTTP client module" "pass"
else
    print_test "HTTP client module" "fail" "Cannot import HTTPRobotClient"
fi

# Check camera modules
if python3 -c "import pyrealsense2" 2>/dev/null; then
    print_test "RealSense library" "pass"
else
    print_test "RealSense library" "warn" "pyrealsense2 not installed (OK if using OAK)"
fi

if python3 -c "import depthai" 2>/dev/null; then
    print_test "OAK library" "pass"
else
    print_test "OAK library" "warn" "depthai not installed (OK if using RealSense)"
fi

# Check other dependencies
if python3 -c "import cv2" 2>/dev/null; then
    print_test "OpenCV" "pass"
else
    print_test "OpenCV" "fail" "opencv-python not installed"
fi

if python3 -c "import numpy" 2>/dev/null; then
    print_test "NumPy" "pass"
else
    print_test "NumPy" "fail" "numpy not installed"
fi

if python3 -c "import scipy" 2>/dev/null; then
    print_test "SciPy" "pass"
else
    print_test "SciPy" "fail" "scipy not installed"
fi

echo ""
echo "3. Checking hardware connections..."

# Test robot server connection
SERVER_URL="http://127.0.0.1:5000"
echo -n "Testing robot server at $SERVER_URL... "
if curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" 2>/dev/null | grep -q "200"; then
    print_test "Robot server" "pass"
else
    print_test "Robot server" "fail" "Server not responding. Run: python franka_server.py"
fi

# Test camera availability
echo ""
echo "4. Checking camera devices..."

# Check RealSense cameras
python3 - <<EOF 2>/dev/null
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        print("REALSENSE_FOUND")
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"   Camera {i}: {name} (Serial: {serial})")
    else:
        print("REALSENSE_NOT_FOUND")
except:
    print("REALSENSE_ERROR")
EOF

RS_RESULT=$?
if [ $RS_RESULT -eq 0 ]; then
    if grep -q "REALSENSE_FOUND" <<< "$(python3 -c 'from dexumi.camera.realsense_camera import get_all_realsense_cameras; cams=get_all_realsense_cameras(); print("REALSENSE_FOUND" if cams else "REALSENSE_NOT_FOUND")' 2>/dev/null)"; then
        print_test "RealSense camera detection" "pass"
    else
        print_test "RealSense camera detection" "warn" "No RealSense cameras found"
    fi
fi

# Check OAK cameras
python3 - <<EOF 2>/dev/null
try:
    from dexumi.camera.oak_camera import get_all_oak_cameras
    cameras = get_all_oak_cameras()
    if cameras:
        print("OAK_FOUND")
        for i, cam_id in enumerate(cameras):
            print(f"   Camera {i}: {cam_id}")
    else:
        print("OAK_NOT_FOUND")
except:
    print("OAK_ERROR")
EOF

OAK_RESULT=$?
if [ $OAK_RESULT -eq 0 ]; then
    if grep -q "OAK_FOUND" <<< "$(python3 -c 'from dexumi.camera.oak_camera import get_all_oak_cameras; cams=get_all_oak_cameras(); print("OAK_FOUND" if cams else "OAK_NOT_FOUND")' 2>/dev/null)"; then
        print_test "OAK camera detection" "pass"
    else
        print_test "OAK camera detection" "warn" "No OAK cameras found"
    fi
fi

echo ""
echo "5. Checking model files..."

# Default model path
MODEL_BASE="/home/gray/Project/DexUMI/data/weight"
if [ -d "$MODEL_BASE" ]; then
    echo "Available models in $MODEL_BASE:"
    for model_dir in "$MODEL_BASE"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            ckpt_count=$(ls -1 "$model_dir"/ckpt_*.pt 2>/dev/null | wc -l)
            if [ $ckpt_count -gt 0 ]; then
                echo -e "   ${GREEN}✓${NC} $model_name ($ckpt_count checkpoints)"
            else
                echo -e "   ${YELLOW}⚠${NC} $model_name (no checkpoints)"
            fi
        fi
    done
else
    print_test "Model directory" "warn" "$MODEL_BASE not found"
fi

echo ""
echo "========================================="
echo "Test Summary:"
echo "  Passed: $TESTS_PASSED"
echo "  Failed: $TESTS_FAILED"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All critical tests passed!${NC}"
    echo ""
    echo "You can now run the evaluation script:"
    echo "  ./eval_xhand_franka.sh"
else
    echo -e "${RED}Some tests failed. Please fix the issues above.${NC}"
fi
echo "========================================="

exit $TESTS_FAILED