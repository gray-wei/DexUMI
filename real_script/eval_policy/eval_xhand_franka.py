import time
from collections import deque

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from dexumi.camera.realsense_camera import RealSenseCamera, get_all_realsense_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.common.utility.matrix import (
    homogeneous_matrix_to_6dof,
    vec6dof_to_homogeneous_matrix,
)
from dexumi.constants import (
    XHAND_HAND_MOTOR_SCALE_FACTOR,
)
from dexumi.data_recording.data_buffer import PoseInterpolator

# Import HTTP control classes
from dexumi.real_env.common.http_client import HTTPRobotClient, HTTPHandClient


def compute_total_force_per_finger(all_fsr_observations):
    """
    Compute the total force for each finger.

    Parameters:
    all_fsr_observations (numpy.ndarray): Array of shape (m, n, 3) where:
        - m is the number of observations
        - n is the number of fingers
        - 3 is the xyz force components

    Returns:
    numpy.ndarray: Array of shape (m, n) with total force magnitude for each finger
    """
    # Calculate the Euclidean norm (magnitude) of the 3D force vector for each finger
    # This computes sqrt(x² + y² + z²) for each finger at each observation
    total_force = np.linalg.norm(all_fsr_observations, axis=2)

    return total_force


# Fixed initial positions (consistent with data collection)
# Initial robot pose from XhandMultimodalCollection.py (7D: xyz + quaternion)
initial_robot_pose = np.array([
    0.5548533772485196, 0.0881488236773295, 0.19591474184161564,
    0.7717268607706337, 0.6356662995622798, 0.01888957126428907, 0.0030318415777171154
])

# Observation and FSR parameters
obs_horizon = 1
binary_cutoff = [10, 10, 10]

# Initial hand position (open position)
initial_hand_pos = np.array([
    0.92755819,
    0.52026953,
    0.22831853,
    0.0707963,
    1.1,
    0.15707963,
    0.95,
    0.12217305,
    1.0392188,
    0.03490659,
    1.0078164,
    0.17453293,
])


@click.command()
@click.option("-f", "--frequency", type=float, default=10, help="Control frequency (Hz)")
@click.option(
    "-mp",
    "--model_path",
    help="Path to the model",
)
@click.option("-ckpt", "--ckpt", type=int, default=600, help="Checkpoint number")
@click.option(
    "-cl", "--camera_latency", type=float, default=0.185, help="Camera latency"
)
@click.option(
    "-hal", "--hand_action_latency", type=float, default=0.3, help="Hand action latency"
)
@click.option(
    "-ral",
    "--robot_action_latency",
    type=float,
    default=0.170,
    help="Robot action latency",
)
@click.option("-eh", "--exec_horizon", type=int, default=8, help="Execution horizon")
def main(
    frequency,
    model_path,
    ckpt,
    camera_latency,
    hand_action_latency,
    robot_action_latency,
    exec_horizon,
):
    # Initialize HTTP clients for robot and hand control
    robot_client = HTTPRobotClient(base_url="http://127.0.0.1:5000")
    dexhand_client = HTTPHandClient(base_url="http://127.0.0.1:5000")
    
    # Initialize RealSense cameras
    all_cameras = get_all_realsense_cameras()
    if len(all_cameras) < 1:
        print("Warning: No RealSense cameras found. Exiting...")
        return
    
    # Use the first available camera for observation
    # Match training format exactly: 424x240 BGR -> center crop to 240x240 (same as XhandMultimodalCollection.py)
    obs_camera = RealSenseCamera(
        camera_name="obs camera",
        device_id=all_cameras[0],
        camera_resolution=(424, 240),  # Exact same as training: 424x240
        enable_depth=False,
        fps=30
    )
    
    # Start camera
    obs_camera.start_streaming()
    

    dt = 1 / frequency
    
    # Main control loop
    while True:
        print("Ready! Starting 20-second inference session...")
            
        # Reset robot to initial position
        print("Moving robot to initial position...")
        # Convert initial pose from 7D (xyz + quat) to 6D (xyz + rotvec) for compatibility
        initial_pose_6d = np.zeros(6)
        initial_pose_6d[:3] = initial_robot_pose[:3]
        initial_pose_6d[3:] = st.Rotation.from_quat(initial_robot_pose[3:]).as_rotvec()
        robot_client.schedule_waypoint(initial_pose_6d, time.time())
        time.sleep(2)  # Wait for robot to reach initial position
        print("Robot at initial position")
        
        # Initialize FSR observations
        fsr_obs = deque(maxlen=obs_horizon)
        fsr_raw_obs = dexhand_client.get_tactile(calc=True)
        # Reshape fsr_raw_obs to add a batch dimension
        fsr_raw_obs = fsr_raw_obs[None, ...]  # This adds a dimension at the start
        fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]
        fsr_value = np.array(fsr_raw_obs[:3])
        print("fsr_value", fsr_value)
        for _ in range(obs_horizon):
            fsr_obs.append(np.zeros(2))

        print(
            "resetting hand----------------------------------------------------------------------------------"
        )

        virtual_hand_pos = initial_hand_pos
        for _ in range(3):
            dexhand_client.schedule_waypoint(
                target_pos=initial_hand_pos,
                target_time=time.time() + 0.05,
            )
            time.sleep(1)
        print(
            "reset done ----------------------------------------------------------------------------------"
        )
        
        # Load policy model
        from dexumi.real_env.real_policy import RealPolicy

        policy = RealPolicy(
            model_path=model_path,
            ckpt=ckpt,
        )

        # Calculate inference parameters
        inference_iter_time = exec_horizon * dt
        inference_fps = 1 / inference_iter_time
        print("inference_fps", inference_fps)
        
        # Start 20-second inference session
        session_start_time = time.time()
        session_duration = 20.0  # 20 seconds
        
        # Policy execution loop
        while time.time() - session_start_time < session_duration:
                with FrameRateContext(frame_rate=inference_fps):
                    # Get observation from camera
                    obs_frame = obs_camera.get_latest_frame()
                    obs_frame_recieved_time = obs_frame.receive_time
                    obs_frame_rgb = obs_frame.rgb.copy()
                    
                    # Note: real_policy.py will handle all image preprocessing
                    # The image should be in BGR format to match training data
                    print(f"Time remaining: {session_duration - (time.time() - session_start_time):.1f}s")
                    if policy.model_cfg.dataset.enable_fsr:
                        print("Using FSR")
                        fsr_raw_obs = dexhand_client.get_tactile(calc=True)
                        print("raw", fsr_raw_obs)
                        # Reshape fsr_raw_obs to add a batch dimension
                        fsr_raw_obs = fsr_raw_obs[
                            None, ...
                        ]  # This adds a dimension at the start
                        fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]
                        fsr_value = np.array(fsr_raw_obs[:3])
                        print("fsr_value", fsr_value)
                        fsr_value = fsr_value.astype(np.float32)
                        # Apply binary cutoff
                        fsr_value_binary = (fsr_value >= binary_cutoff).astype(
                            np.float32
                        )
                        fsr_obs.append(fsr_value_binary)
                    # inference action
                    t_inference = time.monotonic()
                    # camera latency + transfer time
                    print(
                        "t_inference|obs_frame_recieved_time",
                        t_inference,
                        obs_frame_recieved_time,
                    )
                    camera_total_latency = (
                        camera_latency + t_inference - obs_frame_recieved_time
                    )
                    print("camera_total_latency", camera_total_latency)
                    t_actual_inference = t_inference - camera_total_latency
                    
                    # ============ DEBUG SECTION START ============
                    print("\n" + "="*50)
                    print("DEBUG: Input Information")
                    print(f"Image shape: {obs_frame_rgb.shape}, dtype: {obs_frame_rgb.dtype}")
                    print(f"Image range: [{obs_frame_rgb.min():.2f}, {obs_frame_rgb.max():.2f}]")
                    if policy.model_cfg.dataset.enable_fsr:
                        print(f"FSR obs shape: {np.array(list(fsr_obs)).shape}")
                        print(f"FSR obs values: {np.array(list(fsr_obs))[-1]}")  # Last FSR reading
                    
                    action = policy.predict_action(
                        None,
                        np.array(list(fsr_obs)).astype(np.float32)
                        if policy.model_cfg.dataset.enable_fsr
                        else None,
                        obs_frame_rgb[None, ...],  # Use original image, let real_policy.py handle preprocessing
                    )
                    
                    print("\nDEBUG: Raw Action Output")
                    print(f"Action shape: {action.shape}")
                    print(f"Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                    print(f"Action mean/std: [mean={action.mean():.4f}, std={action.std():.4f}]")
                    
                    # convert to abs action
                    relative_pose = action[:, :6]
                    hand_action = action[:, 6:]
                    
                    print("\nDEBUG: Action Components")
                    print(f"Relative pose (first): {relative_pose[0]}")
                    print(f"Hand action (first): {hand_action[0][:4]}...")  # Show first 4 joints
                    print(f"Relative pose norm: {np.linalg.norm(relative_pose[0][:3]):.4f}")  # Position magnitude
                    print(f"Relative rotation norm: {np.linalg.norm(relative_pose[0][3:]):.4f}")  # Rotation magnitude
                    # ============ DEBUG SECTION END ============
                    
                    relative_pose = np.array(
                        [
                            vec6dof_to_homogeneous_matrix(rp[:3], rp[3:])
                            for rp in relative_pose
                        ]
                    )
                    if policy.model_cfg.dataset.relative_hand_action:
                        print("Using relative hand action")
                        # motor_current = dexhand_client.get_pos()
                        motor_current = virtual_hand_pos
                        # print("motor_current", motor_current)
                        motor_current = np.array(motor_current).reshape(1, -1)
                        print(
                            "revaltive hand_action",
                            np.round(
                                hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR, 2
                            ),
                        )
                        hand_action = (
                            motor_current
                            + hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR
                        )
                        offset = np.array([0.0] * 12)
                        hand_action += offset
                    else:
                        print("Not using relative hand action")
                        hand_action = hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR
                        offset = np.array([0.0] * 12)
                        offset[0] = 0.025

                        hand_action += offset

                    # get the robot pose when images were captured
                    robot_frames = robot_client.get_state_history()
                    robot_timestamp = []
                    robot_homogeneous_matrix = []
                    for rf in robot_frames:
                        robot_timestamp.append(rf["receive_time"])
                        # HTTP client returns 7D pose (xyz + quaternion)
                        tcp_pose = rf["state"]["ActualTCPPose"]
                        xyz = tcp_pose[:3]
                        # Convert quaternion to rotation vector for homogeneous matrix
                        if len(tcp_pose) == 7:
                            rotvec = st.Rotation.from_quat(tcp_pose[3:]).as_rotvec()
                        else:
                            rotvec = tcp_pose[3:]  # Already in rotvec format
                        robot_homogeneous_matrix.append(
                            vec6dof_to_homogeneous_matrix(xyz, rotvec)
                        )
                    robot_timestamp = np.array(robot_timestamp)
                    robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)
                    
                    # Interpolate to get pose at inference time
                    robot_pose_interpolator = PoseInterpolator(
                        timestamps=robot_timestamp,
                        homogeneous_matrix=robot_homogeneous_matrix,
                    )
                    aligned_pose = robot_pose_interpolator([t_actual_inference])[0]
                    ee_aligned_pose = homogeneous_matrix_to_6dof(aligned_pose)
                    
                    # Build current end-effector transformation matrix T_BE
                    T_BE = np.eye(4)
                    T_BE[:3, :3] = st.Rotation.from_rotvec(
                        ee_aligned_pose[3:]
                    ).as_matrix()
                    T_BE[:3, -1] = ee_aligned_pose[:3]
                    
                    # Calculate target poses - SIMPLIFIED without T_ET
                    # Since we're directly in end-effector frame, just apply relative transform
                    T_BN = np.zeros_like(relative_pose)
                    for iter_idx in range(len(relative_pose)):
                        # Direct application: T_BN = T_BE @ relative_pose
                        T_BN[iter_idx] = T_BE @ relative_pose[iter_idx]
                    
                    # ============ DEBUG: Target Poses ============
                    print("\nDEBUG: Target Transformation")
                    print(f"Current EE pose: {ee_aligned_pose[:3]}")  # Current position
                    print(f"First target pose: {T_BN[0, :3, -1]}")  # First target position
                    print(f"Position change: {T_BN[0, :3, -1] - ee_aligned_pose[:3]}")  # Delta position
                    # ============ DEBUG END ============
                    # discard actions which in the past
                    n_action = T_BN.shape[0]
                    t_exec = time.monotonic()
                    robot_scheduled = 0
                    hand_scheduled = 0

                    # Process robot waypoints
                    robot_times = t_actual_inference + np.arange(n_action) * dt
                    valid_robot_idx = robot_times >= t_exec + robot_action_latency + dt
                    # convert to global time
                    robot_times = robot_times - time.monotonic() + time.time()
                    for k in np.where(valid_robot_idx)[0]:
                        target_pose = np.zeros(6)
                        target_pose[:3] = T_BN[k, :3, -1]
                        target_pose[3:] = st.Rotation.from_matrix(
                            T_BN[k, :3, :3]
                        ).as_rotvec()
                        robot_client.schedule_waypoint(target_pose, robot_times[k])
                        robot_scheduled += 1

                    # Process hand waypoints
                    hand_times = t_actual_inference + np.arange(n_action) * dt
                    valid_hand_idx = hand_times >= t_exec + hand_action_latency + dt
                    # convert to global time
                    hand_times = hand_times - time.monotonic() + time.time()
                    for k in np.where(valid_hand_idx)[0]:
                        target_hand_action = hand_action[k]
                        dexhand_client.schedule_waypoint(
                            target_hand_action, hand_times[k]
                        )
                        hand_scheduled += 1

                    print(
                        f"Scheduled actions: {robot_scheduled} robot waypoints, {hand_scheduled} hand waypoints"
                    )
                    if len(hand_action) > exec_horizon + 1:
                        virtual_hand_pos = hand_action[exec_horizon + 1]
                    else:
                        virtual_hand_pos = hand_action[-1]
        
        # Session completed, reset to initial positions
        print("20-second session completed. Resetting to initial positions...")
        
        # Reset robot to initial position
        initial_pose_6d = np.zeros(6)
        initial_pose_6d[:3] = initial_robot_pose[:3]
        initial_pose_6d[3:] = st.Rotation.from_quat(initial_robot_pose[3:]).as_rotvec()
        robot_client.schedule_waypoint(initial_pose_6d, time.time())
        
        # Reset hand to initial position
        for _ in range(3):
            dexhand_client.schedule_waypoint(
                target_pos=initial_hand_pos,
                target_time=time.time() + 0.05,
            )
            time.sleep(1)
        
        print("Reset completed. Ready for next session.")
        time.sleep(2)


if __name__ == "__main__":
    main()
