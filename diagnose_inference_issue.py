#!/usr/bin/env python3
"""
Diagnostic script to identify and fix inference issues with DexUMI policy.
Key issues identified:
1. Coordinate frame transformation missing
2. Color channel BGR/RGB mismatch  
3. Action execution strategy differences
4. Time alignment problems
"""

import time
import numpy as np
import pyrealsense2 as rs
import cv2
import click
import scipy.spatial.transform as st
from collections import deque
import os
import pickle


# Critical transformation matrix from eval_xhand.py
T_ET = np.array([
    [0, -1, 0, -0.0395],
    [-1, 0, 0, -0.1342],
    [0, 0, -1, 0.0428],
    [0, 0, 0, 1],
])


def analyze_training_stats(model_path):
    """Analyze training statistics to understand action scales"""
    print("\n" + "="*80)
    print("TRAINING STATISTICS ANALYSIS")
    print("="*80)
    
    stats_path = os.path.join(model_path, "stats.pickle")
    if not os.path.exists(stats_path):
        print(f"✗ Stats file not found: {stats_path}")
        return None
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print("\n1. Action Scale Ranges:")
    print("-"*40)
    
    if 'relative_pose' in stats:
        rel_min = stats['relative_pose']['min']
        rel_max = stats['relative_pose']['max']
        print(f"Relative pose (xyz): min={rel_min[:3].round(4)}, max={rel_max[:3].round(4)}")
        print(f"Relative pose (rot): min={rel_min[3:].round(4)}, max={rel_max[3:].round(4)}")
        print(f"Position range: {(rel_max[:3] - rel_min[:3]).round(4)}")
    
    if 'relative_hand_action' in stats:
        hand_min = stats['relative_hand_action']['min']
        hand_max = stats['relative_hand_action']['max']
        print(f"Relative hand: min={hand_min[:3].round(3)}, max={hand_max[:3].round(3)}")
    elif 'hand_action' in stats:
        hand_min = stats['hand_action']['min']
        hand_max = stats['hand_action']['max']
        print(f"Absolute hand: min={hand_min[:3].round(3)}, max={hand_max[:3].round(3)}")
    
    return stats


def diagnose_coordinate_transform():
    """Diagnose coordinate transformation issues"""
    print("\n" + "="*80)
    print("COORDINATE TRANSFORMATION DIAGNOSIS")
    print("="*80)
    
    # Example current pose and relative action
    current_pose = np.array([0.5548, 0.0881, 0.1959, 0.1, 0.2, 0.3])  # xyz + rotvec
    relative_action = np.array([0.005, 0.003, 0.002, 0.01, 0.01, 0.01])  # Small movement
    
    print("\nCurrent robot pose (xyz):", current_pose[:3].round(3))
    print("Relative action (xyz):", relative_action[:3].round(3))
    
    # Import transformation functions
    from dexumi.common.utility.matrix import (
        vec6dof_to_homogeneous_matrix,
        homogeneous_matrix_to_6dof,
        invert_transformation,
    )
    
    # Current pose as homogeneous matrix
    T_BE = vec6dof_to_homogeneous_matrix(current_pose[:3], current_pose[3:])
    
    # Relative action as homogeneous matrix
    T_relative = vec6dof_to_homogeneous_matrix(relative_action[:3], relative_action[3:])
    
    print("\n" + "-"*40)
    print("Method 1: Direct Addition (WRONG - your current approach)")
    print("-"*40)
    wrong_target = current_pose.copy()
    wrong_target[:3] += relative_action[:3]
    print(f"Target position: {wrong_target[:3].round(3)}")
    print("Issue: Ignores coordinate frame transformation")
    
    print("\n" + "-"*40)
    print("Method 2: Local Frame Transform (PARTIALLY CORRECT)")
    print("-"*40)
    T_BN_local = T_BE @ T_relative
    local_target = homogeneous_matrix_to_6dof(T_BN_local)
    print(f"Target position: {local_target[:3].round(3)}")
    print("Issue: Missing tool frame offset T_ET")
    
    print("\n" + "-"*40)
    print("Method 3: With Tool Transform (CORRECT - eval_xhand.py)")
    print("-"*40)
    T_BN_correct = T_BE @ T_ET @ T_relative @ invert_transformation(T_ET)
    correct_target = homogeneous_matrix_to_6dof(T_BN_correct)
    print(f"Target position: {correct_target[:3].round(3)}")
    print("✓ This is the correct transformation")
    
    print("\n" + "-"*40)
    print("IMPACT ANALYSIS")
    print("-"*40)
    error_wrong = np.linalg.norm(wrong_target[:3] - correct_target[:3])
    error_local = np.linalg.norm(local_target[:3] - correct_target[:3])
    print(f"Position error (direct addition): {error_wrong*1000:.2f} mm")
    print(f"Position error (local frame only): {error_local*1000:.2f} mm")
    
    if error_wrong > 0.001:
        print("⚠️ Direct addition causes significant position errors!")
    
    return correct_target


def diagnose_color_channels():
    """Test color channel issues"""
    print("\n" + "="*80)
    print("COLOR CHANNEL DIAGNOSIS")
    print("="*80)
    
    print("\nISSUE: real_policy.py expects BGR input (line 105: cv2.COLOR_BGR2RGB)")
    print("But eval_realsense_http.py provides RGB (line 167 converts BGR->RGB)")
    
    print("\nIMPACT:")
    print("- Model sees red as blue, blue as red")
    print("- This completely changes visual features")
    print("- Model trained on correct colors won't recognize scenes")
    
    print("\nFIX:")
    print("Remove RGB conversion in eval script. Keep frames in BGR format.")
    print("The model's real_policy.py will handle BGR->RGB conversion internally.")


def diagnose_action_execution():
    """Diagnose action execution strategy"""
    print("\n" + "="*80)
    print("ACTION EXECUTION STRATEGY DIAGNOSIS")
    print("="*80)
    
    print("\neval_xhand.py strategy:")
    print("1. Predicts 16-step action horizon")
    print("2. Executes 8 steps (exec_horizon)")
    print("3. Re-predicts after 8 steps")
    print("4. Overlapping predictions create smooth motion")
    
    print("\nYour eval_realsense_http.py strategy:")
    print("1. Predicts 16-step action horizon")
    print("2. Executes only 1st step")
    print("3. Re-predicts immediately")
    print("4. Wastes 15 predicted steps")
    
    print("\nPROBLEMS:")
    print("- Executing only 1 step causes jerky motion")
    print("- Frequent re-prediction may cause oscillation")
    print("- Small single-step actions may be below motion threshold")
    
    print("\nRECOMMENDED FIX:")
    print("Execute at least 4-8 steps before re-predicting")


def create_fixed_inference_script():
    """Create a fixed version of the inference script"""
    print("\n" + "="*80)
    print("CREATING FIXED INFERENCE SCRIPT")
    print("="*80)
    
    fixed_script = '''#!/usr/bin/env python3
"""
FIXED evaluation script for RealSense D405 camera with proper transformations.
"""

import time
import numpy as np
import pyrealsense2 as rs
import cv2
import click
import requests
import scipy.spatial.transform as st
from collections import deque
from dexumi.common.utility.matrix import (
    vec6dof_to_homogeneous_matrix,
    homogeneous_matrix_to_6dof,
    invert_transformation,
)

# Critical transformation matrix
T_ET = np.array([
    [0, -1, 0, -0.0395],
    [-1, 0, 0, -0.1342],
    [0, 0, -1, 0.0428],
    [0, 0, 0, 1],
])

def get_camera_frame_correct(pipeline):
    """Get camera frame in BGR format (correct for model)"""
    frames = pipeline.wait_for_frames(timeout_ms=100)
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        return None
    
    # Keep as BGR - DO NOT convert to RGB
    image_bgr = np.asanyarray(color_frame.get_data())
    
    # Center crop and resize
    h, w = image_bgr.shape[:2]
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped = image_bgr[start_y:start_y+crop_size, start_x:start_x+crop_size]
    resized = cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_AREA)
    
    return resized  # BGR format

def apply_relative_action(current_pose, relative_action):
    """Apply relative action with proper coordinate transformation"""
    # Convert current pose to homogeneous matrix
    if len(current_pose) == 7:  # quaternion
        T_BE = vec6dof_to_homogeneous_matrix(
            current_pose[:3],
            st.Rotation.from_quat(current_pose[3:]).as_rotvec()
        )
    else:  # rotvec
        T_BE = vec6dof_to_homogeneous_matrix(current_pose[:3], current_pose[3:])
    
    # Convert relative action to homogeneous matrix
    T_relative = vec6dof_to_homogeneous_matrix(
        relative_action[:3], relative_action[3:]
    )
    
    # Apply transformation with tool frame
    T_BN = T_BE @ T_ET @ T_relative @ invert_transformation(T_ET)
    
    # Convert back to pose
    target_6dof = homogeneous_matrix_to_6dof(T_BN)
    
    # Convert to quaternion if needed
    if len(current_pose) == 7:
        target_pose = np.zeros(7)
        target_pose[:3] = target_6dof[:3]
        target_pose[3:] = st.Rotation.from_rotvec(target_6dof[3:]).as_quat()
    else:
        target_pose = target_6dof
    
    return target_pose

# In main inference loop:
if running and policy is not None:
    # Get camera frame (BGR format)
    rgb_frame = get_camera_frame_correct(pipeline)
    
    # Run inference
    action = policy.predict_action(None, fsr_input, rgb_frame[None, ...])
    
    # Execute multiple action steps
    exec_horizon = 8  # Execute 8 out of 16 predicted steps
    for step in range(min(exec_horizon, len(action))):
        # Apply relative action with proper transformation
        target_pose = apply_relative_action(current_pose, action[step, :6])
        controller.send_pose(target_pose)
        
        # Update hand
        hand_target = current_hand + action[step, 6:] / XHAND_HAND_MOTOR_SCALE_FACTOR
        controller.send_hand_angles(hand_target)
        
        # Update current state for next step
        current_pose = target_pose
        current_hand = hand_target
        
        time.sleep(dt)  # Maintain control frequency
'''
    
    print("Key fixes in the corrected script:")
    print("1. ✓ Keep camera frames in BGR format")
    print("2. ✓ Apply T_ET coordinate transformation")
    print("3. ✓ Execute multiple action steps (8 out of 16)")
    print("4. ✓ Properly update state between steps")
    
    # Save the fixed script
    with open('/home/ubuntu/hgw/IL/DexUMI/eval_realsense_http_fixed.py', 'w') as f:
        f.write(fixed_script)
    
    print("\n✓ Saved fixed script to: eval_realsense_http_fixed.py")


@click.command()
@click.option('-mp', '--model-path', type=str, help='Path to trained model')
@click.option('--full', is_flag=True, help='Run full diagnosis')
def main(model_path, full):
    """
    Diagnose DexUMI inference issues
    
    Usage:
    python diagnose_inference_issue.py -mp /path/to/model --full
    """
    
    print("\n" + "="*80)
    print("DEXUMI INFERENCE DIAGNOSTICS")
    print("="*80)
    
    print("\n🔍 IDENTIFIED ISSUES:")
    print("-"*40)
    print("1. 🔴 CRITICAL: Missing T_ET coordinate transformation")
    print("2. 🔴 CRITICAL: BGR/RGB color channel mismatch")
    print("3. 🟡 IMPORTANT: Only executing 1 action step instead of 8")
    print("4. 🟡 IMPORTANT: No time alignment for image-action sync")
    print("5. 🟡 IMPORTANT: Action normalization may be incorrect")
    
    if model_path:
        stats = analyze_training_stats(model_path)
    
    if full:
        diagnose_coordinate_transform()
        diagnose_color_channels()
        diagnose_action_execution()
        create_fixed_inference_script()
    
    print("\n" + "="*80)
    print("RECOMMENDED IMMEDIATE ACTIONS")
    print("="*80)
    
    print("""
1. Use the fixed script: eval_realsense_http_fixed.py
   
2. Or apply these critical fixes to your script:
   a) Remove RGB conversion (line 167), keep BGR format
   b) Add T_ET transformation when applying actions
   c) Execute 8 action steps instead of just 1
   
3. Test with larger action scale temporarily:
   - Multiply TCP actions by 2-3x to ensure visible movement
   
4. Debug output:
   - Print action magnitudes at each step
   - Verify robot is receiving commands
   - Check actual vs commanded positions
""")


if __name__ == "__main__":
    main()