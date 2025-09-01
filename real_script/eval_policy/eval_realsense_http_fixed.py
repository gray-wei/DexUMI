#!/usr/bin/env python3
"""
FIXED evaluation script for RealSense D405 camera with HTTP control.
Fixes critical issues:
1. Coordinate frame transformation (T_ET)
2. BGR/RGB color channel mismatch
3. Action execution strategy (8 steps instead of 1)
4. Proper state tracking between steps
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
from dexumi.constants import XHAND_HAND_MOTOR_SCALE_FACTOR


# Critical transformation matrix from eval_xhand.py
T_ET = np.array([
    [0, -1, 0, -0.0395],
    [-1, 0, 0, -0.1342],
    [0, 0, -1, 0.0428],
    [0, 0, 0, 1],
])

# Initial robot pose (from XhandMultimodalCollection.py)
INIT_ROBOT_POSE = np.array([
    0.5548533772485196, 0.0881488236773295, 0.19591474184161564,
    0.7717268607706337, 0.6356662995622798, 0.01888957126428907, 0.0030318415777171154
])

# Initial hand pose (open position from eval_xhand.py)
INIT_HAND_POSE = np.array([
    0.92755819, 0.52026953, 0.22831853, 0.0707963,
    1.1, 0.15707963, 0.95, 0.12217305,
    1.0392188, 0.03490659, 1.0078164, 0.17453293,
])


class SimpleHTTPController:
    """Simple HTTP interface for robot and hand control"""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.url = base_url
        
    def reset_robot(self):
        """Reset robot to initial pose"""
        try:
            response = requests.post(
                f"{self.url}/pose",
                json={"arr": INIT_ROBOT_POSE.tolist()},
                timeout=2.0
            )
            if response.status_code == 200:
                print("✓ Robot reset to initial pose")
                return True
        except Exception as e:
            print(f"✗ Failed to reset robot: {e}")
        return False
    
    def reset_hand(self):
        """Reset hand to initial position"""
        try:
            # Send specific hand pose
            response = requests.post(
                f"{self.url}/hand_pose",
                json={"arr": INIT_HAND_POSE.tolist()},
                timeout=2.0
            )
            if response.status_code == 200:
                print("✓ Hand reset to initial position")
                return True
        except Exception as e:
            print(f"✗ Failed to reset hand: {e}")
        return False
    
    def get_state(self):
        """Get current robot state"""
        try:
            response = requests.post(f"{self.url}/getstate", timeout=1.0)
            if response.status_code == 200:
                data = response.json()
                return {
                    'tcp_pose': np.array(data["pose"]),  # 7D: xyz + quaternion
                    'hand_angles': np.array(data["gripper_pos"]),  # 12D
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"Error getting state: {e}")
        return None
    
    def send_pose(self, pose):
        """Send TCP pose command (7D: xyz + quaternion)"""
        try:
            # Ensure 7D format
            if len(pose) == 6:  # Convert from 6D (xyz + rotvec) to 7D (xyz + quat)
                xyz = pose[:3]
                quat = st.Rotation.from_rotvec(pose[3:]).as_quat()
                pose = np.concatenate([xyz, quat])
            
            response = requests.post(
                f"{self.url}/pose",
                json={"arr": pose.tolist()},
                timeout=1.0
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending pose: {e}")
        return False
    
    def send_hand_angles(self, angles):
        """Send hand joint angles"""
        try:
            response = requests.post(
                f"{self.url}/hand_pose",
                json={"arr": angles.tolist()},
                timeout=1.0
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending hand angles: {e}")
        return False
    
    def get_tactile(self):
        """Get tactile/FSR data"""
        try:
            response = requests.post(f"{self.url}/get_handtactile", timeout=1.0)
            if response.status_code == 200:
                return np.array(response.json()["tactile_data"])
        except Exception as e:
            print(f"Error getting tactile: {e}")
        return np.zeros((5, 3))


def setup_realsense_camera():
    """Setup RealSense D405 camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure for 640x480 RGB at 30 fps (D405 native resolution)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start pipeline
    pipeline.start(config)
    
    # Allow auto-exposure to settle
    for _ in range(30):
        pipeline.wait_for_frames()
    
    print("✓ RealSense D405 camera initialized")
    return pipeline


def get_camera_frame(pipeline):
    """Get frame from RealSense camera and crop to 240x240 - KEEP BGR FORMAT"""
    try:
        frames = pipeline.wait_for_frames(timeout_ms=100)
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
        
        # Get as BGR (native format) - DO NOT CONVERT TO RGB
        image = np.asanyarray(color_frame.get_data())  # BGR format, shape: (480, 640, 3)
        
        # Center crop to square (480x480)
        h, w = image.shape[:2]
        crop_size = min(h, w)  # 480
        start_x = (w - crop_size) // 2  # (640-480)//2 = 80
        start_y = (h - crop_size) // 2  # 0
        cropped = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Resize to 240x240 (matching training data)
        resized = cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_AREA)
        
        # KEEP AS BGR - Model expects BGR and converts internally
        return resized
    except Exception as e:
        print(f"Camera frame error: {e}")
        return None


def apply_relative_action_with_transform(current_pose, relative_action):
    """Apply relative action with proper T_ET coordinate transformation"""
    # Convert current pose (7D quaternion) to homogeneous matrix
    T_BE = vec6dof_to_homogeneous_matrix(
        current_pose[:3],
        st.Rotation.from_quat(current_pose[3:]).as_rotvec()
    )
    
    # Convert relative action to homogeneous matrix
    T_relative = vec6dof_to_homogeneous_matrix(
        relative_action[:3], relative_action[3:]
    )
    
    # Apply transformation WITH tool frame (T_ET)
    T_BN = T_BE @ T_ET @ T_relative @ invert_transformation(T_ET)
    
    # Convert back to 7D pose (xyz + quat)
    target_6dof = homogeneous_matrix_to_6dof(T_BN)
    target_pose = np.zeros(7)
    target_pose[:3] = target_6dof[:3]
    target_pose[3:] = st.Rotation.from_rotvec(target_6dof[3:]).as_quat()
    
    return target_pose


@click.command()
@click.option('-mp', '--model-path', type=str, help='Path to trained model')
@click.option('-ckpt', '--checkpoint', type=int, default=600, help='Model checkpoint')
@click.option('-f', '--frequency', type=float, default=10, help='Control frequency (Hz)')
@click.option('-eh', '--exec-horizon', type=int, default=8, help='Execution horizon (steps to execute)')
@click.option('--debug', is_flag=True, help='Enable debug output')
def main(model_path, checkpoint, frequency, exec_horizon, debug):
    """
    FIXED evaluation script with proper coordinate transforms and color handling.
    
    Usage:
    1. Start robot server: ./launch_right_server.sh
    2. Run: python eval_realsense_http_fixed.py -mp /path/to/model -ckpt 600
    
    Keyboard controls:
    - SPACE: Start/stop policy execution
    - R: Reset to initial pose
    - Q: Quit current episode
    - ESC: Exit program
    """
    
    print("\n=== DexUMI Evaluation (FIXED) ===\n")
    print("Fixes applied:")
    print("✓ T_ET coordinate transformation")
    print("✓ BGR color format (no RGB conversion)")
    print(f"✓ Multi-step execution (horizon={exec_horizon})")
    print("✓ Proper state tracking\n")
    
    # Initialize controller
    controller = SimpleHTTPController()
    
    # Test connection
    state = controller.get_state()
    if state is None:
        print("✗ Cannot connect to robot server at http://127.0.0.1:5000")
        print("  Please run: ./launch_right_server.sh")
        return
    
    print(f"✓ Connected to robot server")
    print(f"  Current position: {state['tcp_pose'][:3]}")
    
    # Setup camera
    pipeline = setup_realsense_camera()
    
    # Load policy model
    if model_path:
        from dexumi.real_env.real_policy import RealPolicy
        
        policy = RealPolicy(model_path=model_path, ckpt=checkpoint)
        print(f"✓ Loaded model from {model_path}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Execution horizon: {exec_horizon} steps")
        
        # Check if FSR is enabled in model
        use_fsr = hasattr(policy.model_cfg.dataset, 'enable_fsr') and policy.model_cfg.dataset.enable_fsr
        relative_hand = hasattr(policy.model_cfg.dataset, 'relative_hand_action') and \
                       policy.model_cfg.dataset.relative_hand_action
        
        print(f"  FSR enabled: {use_fsr}")
        print(f"  Relative hand action: {relative_hand}")
    else:
        policy = None
        use_fsr = False
        relative_hand = False
        print("✗ No model specified, manual control only")
    
    dt = 1.0 / frequency
    print(f"\nControl frequency: {frequency} Hz (dt={dt:.3f}s)")
    
    # Test camera
    print("\nTesting camera...")
    test_frame = get_camera_frame(pipeline)
    if test_frame is None:
        print("✗ Failed to get camera frame")
        pipeline.stop()
        return
    print(f"✓ Camera test successful (frame shape: {test_frame.shape}, dtype: {test_frame.dtype})")
    
    try:
        while True:
            print("\n" + "="*50)
            print("NEW EPISODE")
            print("="*50)
            
            # Reset robot and hand to initial pose
            print("\nResetting system...")
            controller.reset_robot()
            controller.reset_hand()
            time.sleep(2.0)  # Wait for reset to complete
            
            print("\nControls:")
            print("  SPACE - Start/stop policy execution")
            print("  R     - Reset to initial pose")
            print("  Q     - Quit episode")
            print("  ESC   - Exit program")
            
            # Manual positioning phase
            print("\n--- Manual Positioning Phase ---")
            print("Position the robot manually, then press SPACE to start policy")
            
            running = False
            episode_active = True
            
            # Initialize state tracking
            current_hand = INIT_HAND_POSE.copy()
            
            # Initialize FSR buffer if needed
            if use_fsr:
                fsr_buffer = deque(maxlen=1)
                for _ in range(1):
                    fsr_buffer.append(np.zeros(3))
            
            # Track inference timing
            last_inference_time = 0
            inference_interval = exec_horizon * dt  # Re-predict after executing N steps
            action_queue = []
            action_step = 0
            
            # Main control loop
            while episode_active:
                # Get camera frame (BGR format!)
                bgr_frame = get_camera_frame(pipeline)
                if bgr_frame is None:
                    continue
                
                # Display frame
                display_frame = bgr_frame.copy()
                
                # Add status text
                status = "POLICY RUNNING" if running else "MANUAL CONTROL"
                color = (0, 255, 0) if running else (0, 165, 255)
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show action info if running
                if running and len(action_queue) > 0:
                    cv2.putText(display_frame, f"Step: {action_step+1}/{len(action_queue)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Show FSR values if enabled
                if use_fsr and running:
                    tactile = controller.get_tactile()
                    force = np.linalg.norm(tactile, axis=1)[:3]
                    cv2.putText(display_frame, f"FSR: {force[0]:.1f} {force[1]:.1f} {force[2]:.1f}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("RealSense D405 (FIXED)", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - toggle policy
                    if policy is not None:
                        running = not running
                        if running:
                            print("\n▶ Policy execution started")
                            last_inference_time = 0  # Force immediate inference
                            action_queue = []
                            action_step = 0
                        else:
                            print("\n⏸ Policy execution paused")
                
                elif key == ord('r'):  # R - reset
                    print("\nResetting...")
                    running = False
                    controller.reset_robot()
                    controller.reset_hand()
                    current_hand = INIT_HAND_POSE.copy()
                    time.sleep(2.0)
                
                elif key == ord('q'):  # Q - quit episode
                    print("\nEnding episode...")
                    episode_active = False
                
                elif key == 27:  # ESC - exit program
                    print("\nExiting program...")
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    return
                
                # Execute policy if running
                if running and policy is not None:
                    current_time = time.time()
                    
                    # Check if we need new inference
                    if action_step >= len(action_queue) or \
                       (current_time - last_inference_time) >= inference_interval:
                        
                        # Get current state for reference
                        state = controller.get_state()
                        if state is None:
                            continue
                        
                        # Get FSR data if needed
                        fsr_input = None
                        if use_fsr:
                            tactile = controller.get_tactile()
                            force = np.linalg.norm(tactile, axis=1)[:3]
                            # Apply binary threshold
                            binary_cutoff = [10, 10, 10]
                            fsr_binary = (force >= binary_cutoff).astype(np.float32)
                            fsr_buffer.append(fsr_binary)
                            fsr_input = np.array(list(fsr_buffer)).astype(np.float32)
                        
                        # Run policy inference (BGR frame!)
                        action = policy.predict_action(
                            None,  # No proprioception
                            fsr_input,
                            bgr_frame[None, ...]  # Add batch dimension
                        )
                        
                        # Store actions and reset counter
                        action_queue = action[:exec_horizon]  # Take first N steps
                        action_step = 0
                        last_inference_time = current_time
                        
                        if debug:
                            print(f"\n[Inference] Generated {len(action_queue)} actions")
                            print(f"  First TCP action: {action_queue[0, :3].round(4)}")
                            print(f"  Action magnitude: {np.linalg.norm(action_queue[0, :3]):.4f}")
                    
                    # Execute current action step
                    if action_step < len(action_queue):
                        # Get current state
                        state = controller.get_state()
                        if state is None:
                            continue
                        
                        current_action = action_queue[action_step]
                        
                        # Apply TCP action with T_ET transformation
                        target_pose = apply_relative_action_with_transform(
                            state['tcp_pose'], 
                            current_action[:6]
                        )
                        
                        # Apply hand action
                        if relative_hand:
                            # Relative action
                            hand_target = current_hand + current_action[6:] / XHAND_HAND_MOTOR_SCALE_FACTOR
                        else:
                            # Absolute action
                            hand_target = current_action[6:] / XHAND_HAND_MOTOR_SCALE_FACTOR
                        
                        # Send commands
                        controller.send_pose(target_pose)
                        controller.send_hand_angles(hand_target)
                        
                        # Update tracking
                        current_hand = hand_target
                        action_step += 1
                        
                        if debug and action_step == 1:
                            print(f"[Step {action_step}] Pose: {target_pose[:3].round(3)}")
                
                # Maintain control frequency
                time.sleep(dt)
            
            cv2.destroyAllWindows()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCleanup complete")


if __name__ == "__main__":
    main()