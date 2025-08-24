"""
Evaluation script using HTTP control interface.
This script communicates with franka_server.py for robot and hand control.
"""

import os
import time
from collections import deque

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from dexumi.camera.camera import FrameData
from dexumi.camera.iphone_camera import IphoneCamera
from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras
from dexumi.common.frame_manager import FrameRateContext
from dexumi.common.precise_sleep import precise_wait
from dexumi.common.utility.matrix import (
    convert_homogeneous_matrix,
    homogeneous_matrix_to_6dof,
    invert_transformation,
    relative_transformation,
    vec6dof_to_homogeneous_matrix,
    visualize_multiple_frames_and_points,
)
from dexumi.common.utility.video import extract_frames_videos
from dexumi.constants import XHAND_HAND_MOTOR_SCALE_FACTOR
from dexumi.data_recording import VideoRecorder
from dexumi.data_recording.data_buffer import PoseInterpolator
from dexumi.data_recording.numeric_recorder import NumericRecorder
from dexumi.data_recording.record_manager import RecorderManager
from dexumi.real_env.common.http_client import HTTPRobotClient, HTTPHandClient
from dexumi.real_env.spacemouse import Spacemouse


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
    total_force = np.linalg.norm(all_fsr_observations, axis=2)
    return total_force


# Hand-eye calibration matrix
x_offset = -0.0395
y_offset = -0.1342
z_offset = 0.0428
T_ET = np.array(
    [
        [0, -1, 0, x_offset],
        [-1, 0, 0, y_offset],
        [0, 0, -1, z_offset],
        [0, 0, 0, 1],
    ]
)

obs_horizon = 1
binary_cutoff = [10, 10, 10]
initial_hand_pos = np.array(
    [
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
    ]
)


@click.command()
@click.option("-ms", "--max_pos_speed", type=float, default=0.1)
@click.option("-mr", "--max_rot_speed", type=float, default=0.6)
@click.option("-f", "--frequency", type=float, default=10)
@click.option("-rc", "--enable_record_camera", is_flag=True, help="Enable record camera")
@click.option("-mp", "--model_path", help="Path to the model")
@click.option("-ckpt", "--ckpt", type=int, default=600, help="Checkpoint number")
@click.option("-cl", "--camera_latency", type=float, default=0.185, help="Camera latency")
@click.option("-hal", "--hand_action_latency", type=float, default=0.3, help="Hand action latency")
@click.option("-ral", "--robot_action_latency", type=float, default=0.170, help="Robot action latency")
@click.option("-eh", "--exec_horizon", type=int, default=8, help="Execution horizon")
@click.option("-vp", "--video_record_path", type=str, default="video_record", help="Path to save video recordings")
@click.option("-mep", "--match_episode_path", type=str, default=None, help="Path to match episode folder")
@click.option("-hu", "--http_url", type=str, default="http://127.0.0.1:5000", help="HTTP server URL")
@click.option("-ht", "--http_timeout", type=float, default=1.0, help="HTTP request timeout")
def main(
    frequency,
    max_pos_speed,
    max_rot_speed,
    enable_record_camera,
    model_path,
    ckpt,
    camera_latency,
    hand_action_latency,
    robot_action_latency,
    exec_horizon,
    video_record_path,
    match_episode_path,
    http_url,
    http_timeout,
):
    """
    Main evaluation loop using HTTP control interface.
    
    Before running this script, make sure franka_server.py is running:
    cd /home/ubuntu/hly/gray_dex_dp/serl_robot_infra/robot_servers
    ./launch_right_server.sh
    """
    
    print(f"Connecting to HTTP server at {http_url}")
    
    # Initialize HTTP clients
    robot_client = HTTPRobotClient(base_url=http_url, timeout=http_timeout)
    dexhand_client = HTTPHandClient(base_url=http_url, timeout=http_timeout)
    
    # Test connection
    try:
        state = robot_client.get_state()
        print("Successfully connected to robot server")
        print(f"Current robot pose: {state['state']['ActualTCPPose'][:3]}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        print("Make sure franka_server.py is running")
        return
    
    # Initialize cameras
    all_cameras = get_all_oak_cameras()
    if len(all_cameras) < 1:
        print("No OAK cameras found!")
        return
        
    obs_camera = OakCamera("obs camera", device_id=all_cameras[0])
    camera_sources = [obs_camera]
    
    if enable_record_camera and len(all_cameras) > 1:
        record_camera = OakCamera("record camera", device_id=all_cameras[1])
        camera_sources.append(record_camera)
    
    # Initialize video recorder
    video_recorder = VideoRecorder(
        record_fps=45,
        stream_fps=60,
        video_record_path=video_record_path,
        camera_sources=camera_sources,
        frame_data_class=FrameData,
        verbose=False,
    )
    
    recorder_manager = RecorderManager(
        recorders=[video_recorder],
        verbose=False,
    )
    recorder_manager.start_streaming()
    
    dt = 1 / frequency
    match_episode_folder = match_episode_path
    
    with Spacemouse() as sm:
        while True:
            print("Ready! Press 'q' to quit, 'x' to exit completely")
            
            # Load match episode if provided
            if match_episode_folder is not None:
                print(f"Extracting frames from match episode {recorder_manager.episode_id}")
                match_episode = extract_frames_videos(
                    os.path.join(
                        match_episode_folder,
                        f"episode_{recorder_manager.episode_id}/camera_1.mp4",
                    ),
                    BGR2RGB=True,
                )
                match_initial_frame = match_episode[0]
            else:
                print("No match episode folder provided")
                match_initial_frame = None
            
            command_latency = dt / 2
            state = robot_client.get_state()
            target_pose = state["state"]["TargetTCPPose"]
            recived_time = state["receive_time"]
            t_start = time.monotonic()
            print(f"State delay: {time.time() - recived_time:.3f}s")
            
            # Manual control loop with spacemouse
            iter_idx = 0
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt
                
                precise_wait(t_sample)
                record_frame = recorder_manager.get_latest_frames()
                
                # Handle frames based on enable_record_camera flag
                if enable_record_camera and "record camera" in record_frame:
                    video_frame = record_frame["record camera"][-1]
                    viz_frame = video_frame.rgb.copy()
                
                obs_frame = record_frame["obs camera"][-1]
                
                # Overlay match_initial_frame and viz_frame
                if match_initial_frame is not None and enable_record_camera:
                    alpha = 0.5  # Transparency factor
                    overlay = cv2.addWeighted(
                        match_initial_frame, alpha, viz_frame, 1 - alpha, 0
                    )
                    cv2.putText(
                        overlay,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Overlay", overlay)
                elif enable_record_camera and "viz_frame" in locals():
                    cv2.putText(
                        viz_frame,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("VIZ", viz_frame)
                
                if obs_frame is not None:
                    cv2.imshow("RGB", obs_frame.rgb)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("x"):
                        exit()
                
                # Get spacemouse input
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                
                drot = st.Rotation.from_euler("xyz", drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()
                
                # Send command to robot via HTTP
                robot_client.schedule_waypoint(
                    target_pose, t_command_target - time.monotonic() + time.time()
                )
                
                precise_wait(t_cycle_end)
                iter_idx += 1
            
            cv2.destroyAllWindows()
            
            # Get robot state history
            robot_frames = robot_client.receive_data()
            if robot_frames:
                robot_timestamp = []
                robot_homogeneous_matrix = []
                for rf in robot_frames:
                    robot_timestamp.append(rf["receive_time"])
                    robot_homogeneous_matrix.append(
                        vec6dof_to_homogeneous_matrix(
                            rf["state"]["ActualTCPPose"][:3],
                            rf["state"]["ActualTCPPose"][3:],
                        )
                    )
                robot_timestamp = np.array(robot_timestamp)
                robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)
                t_now = time.time()
                print(
                    f"Robot data: min={robot_timestamp.min():.2f}, "
                    f"max={robot_timestamp.max():.2f}, "
                    f"now={t_now:.2f}, delay={t_now - robot_timestamp.max():.3f}s"
                )
            
            # Initialize FSR observations
            fsr_obs = deque(maxlen=obs_horizon)
            fsr_raw_obs = dexhand_client.get_tactile(calc=True)
            # Reshape fsr_raw_obs to add a batch dimension
            fsr_raw_obs = fsr_raw_obs[None, ...]  # This adds a dimension at the start
            fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]
            fsr_value = np.array(fsr_raw_obs[:3])
            print("FSR value:", fsr_value)
            
            for _ in range(obs_horizon):
                fsr_obs.append(np.zeros(2))
            
            # Reset hand position
            print("Resetting hand position...")
            virtual_hand_pos = initial_hand_pos
            for i in range(3):
                dexhand_client.schedule_waypoint(
                    target_pos=initial_hand_pos,
                    target_time=time.time() + 0.05,
                )
                time.sleep(1)
            print("Hand reset complete")
            
            # Load policy model
            from dexumi.real_env.real_policy import RealPolicy
            
            if model_path is None:
                print("No model path provided, skipping policy inference")
                continue
                
            policy = RealPolicy(
                model_path=model_path,
                ckpt=ckpt,
            )
            
            if recorder_manager.reset_episode_recording():
                click.echo("Starting recording...")
                recorder_manager.start_recording()
            
            # Policy inference loop
            inference_iter_time = exec_horizon * dt
            inference_fps = 1 / inference_iter_time
            print(f"Inference FPS: {inference_fps}")
            
            while True:
                with FrameRateContext(frame_rate=inference_fps):
                    # Gather observation
                    record_frame = recorder_manager.get_latest_frames()
                    
                    if enable_record_camera and "record camera" in record_frame:
                        video_frame = record_frame["record camera"][-1]
                        viz_frame = video_frame.rgb.copy()
                        cv2.putText(
                            viz_frame,
                            f"Episode: {recorder_manager.episode_id}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    
                    obs_frame = record_frame["obs camera"][-1]
                    obs_frame_recieved_time = obs_frame.receive_time
                    obs_frame_rgb = obs_frame.rgb.copy()
                    
                    cv2.putText(
                        obs_frame_rgb,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    
                    if policy.model_cfg.dataset.enable_fsr:
                        # Draw FSR values on frame
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR1: {fsr_value[0]:.0f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR2: {fsr_value[1]:.0f}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        # Draw binary cutoff values
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR1 Binary: {int(fsr_value[0] > binary_cutoff[0])}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR2 Binary: {int(fsr_value[1] > binary_cutoff[1])}",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR3 Binary: {int(fsr_value[2] > binary_cutoff[2])}",
                            (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    
                    cv2.imshow("obs frame", obs_frame_rgb)
                    if enable_record_camera and "viz_frame" in locals():
                        cv2.imshow("record frame", viz_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        if recorder_manager.stop_recording():
                            recorder_manager.save_recordings()
                        cv2.destroyAllWindows()
                        break
                    elif key == ord("a"):
                        if recorder_manager.stop_recording():
                            recorder_manager.clear_recording()
                        break
                    
                    # Update FSR observations
                    if policy.model_cfg.dataset.enable_fsr:
                        print("Using FSR")
                        fsr_raw_obs = dexhand_client.get_tactile(calc=True)
                        print("Raw FSR:", fsr_raw_obs)
                        # Reshape fsr_raw_obs to add a batch dimension
                        fsr_raw_obs = fsr_raw_obs[None, ...]
                        fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]
                        fsr_value = np.array(fsr_raw_obs[:3])
                        print("FSR value:", fsr_value)
                        fsr_value = fsr_value.astype(np.float32)
                        # Apply binary cutoff
                        fsr_value_binary = (fsr_value >= binary_cutoff).astype(np.float32)
                        fsr_obs.append(fsr_value_binary)
                    
                    # Inference action
                    t_inference = time.monotonic()
                    print(
                        f"t_inference: {t_inference:.3f}, "
                        f"obs_frame_received: {obs_frame_recieved_time:.3f}"
                    )
                    camera_total_latency = (
                        camera_latency + t_inference - obs_frame_recieved_time
                    )
                    print(f"Camera total latency: {camera_total_latency:.3f}s")
                    t_actual_inference = t_inference - camera_total_latency
                    
                    # Predict action
                    action = policy.predict_action(
                        None,
                        np.array(list(fsr_obs)).astype(np.float32)
                        if policy.model_cfg.dataset.enable_fsr
                        else None,
                        obs_frame.rgb[None, ...],
                    )
                    
                    # Convert to absolute action
                    relative_pose = action[:, :6]
                    hand_action = action[:, 6:]
                    relative_pose = np.array(
                        [
                            vec6dof_to_homogeneous_matrix(rp[:3], rp[3:])
                            for rp in relative_pose
                        ]
                    )
                    
                    # Handle hand action
                    if policy.model_cfg.dataset.relative_hand_action:
                        print("Using relative hand action")
                        motor_current = virtual_hand_pos
                        motor_current = np.array(motor_current).reshape(1, -1)
                        print(
                            "Relative hand_action:",
                            np.round(hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR, 2),
                        )
                        hand_action = (
                            motor_current + hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR
                        )
                        offset = np.array([0.0] * 12)
                        hand_action += offset
                    else:
                        print("Not using relative hand action")
                        hand_action = hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR
                        offset = np.array([0.0] * 12)
                        offset[0] = 0.025
                        hand_action += offset
                    
                    # Get robot pose when images were captured
                    robot_frames = robot_client.get_state_history()
                    if robot_frames:
                        robot_timestamp = []
                        robot_homogeneous_matrix = []
                        for rf in robot_frames:
                            robot_timestamp.append(rf["receive_time"])
                            robot_homogeneous_matrix.append(
                                vec6dof_to_homogeneous_matrix(
                                    rf["state"]["ActualTCPPose"][:3],
                                    rf["state"]["ActualTCPPose"][3:],
                                )
                            )
                        robot_timestamp = np.array(robot_timestamp)
                        robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)
                        
                        # Interpolate robot pose
                        robot_pose_interpolator = PoseInterpolator(
                            timestamps=robot_timestamp,
                            homogeneous_matrix=robot_homogeneous_matrix,
                        )
                        aligned_pose = robot_pose_interpolator([t_actual_inference])[0]
                        ee_aligned_pose = homogeneous_matrix_to_6dof(aligned_pose)
                        
                        # Transform relative to absolute pose
                        T_BE = np.eye(4)
                        T_BE[:3, :3] = st.Rotation.from_rotvec(ee_aligned_pose[3:]).as_matrix()
                        T_BE[:3, -1] = ee_aligned_pose[:3]
                        T_BN = np.zeros_like(relative_pose)
                        for iter_idx in range(len(relative_pose)):
                            T_BN[iter_idx] = (
                                T_BE
                                @ T_ET
                                @ relative_pose[iter_idx]
                                @ invert_transformation(T_ET)
                            )
                        
                        # Schedule waypoints
                        n_action = T_BN.shape[0]
                        t_exec = time.monotonic()
                        robot_scheduled = 0
                        hand_scheduled = 0
                        
                        # Process robot waypoints
                        robot_times = t_actual_inference + np.arange(n_action) * dt
                        valid_robot_idx = robot_times >= t_exec + robot_action_latency + dt
                        # Convert to global time
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
                        # Convert to global time
                        hand_times = hand_times - time.monotonic() + time.time()
                        for k in np.where(valid_hand_idx)[0]:
                            target_hand_action = hand_action[k]
                            dexhand_client.schedule_waypoint(target_hand_action, hand_times[k])
                            hand_scheduled += 1
                        
                        print(
                            f"Scheduled actions: {robot_scheduled} robot waypoints, "
                            f"{hand_scheduled} hand waypoints"
                        )
                        virtual_hand_pos = hand_action[min(exec_horizon + 1, len(hand_action) - 1)]


if __name__ == "__main__":
    main()