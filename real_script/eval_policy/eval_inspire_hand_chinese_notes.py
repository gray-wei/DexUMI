# 导入必要的库和模块
import os
import time
from collections import deque

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

# 导入DexUMI项目相关的模块
from dexumi.camera.camera import FrameData
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
from dexumi.common.utility.video import (
    extract_frames_videos,
)
from dexumi.constants import (
    INSPIRE_HAND_MOTOR_SCALE_FACTOR,
    INSPIRE_PER_FINGER_MOTOR_ADJUST_ABS_TOOL_INF,
    INSPIRE_PER_FINGER_MOTOR_ADJUST_EGG_INF,
    INSPIRE_PER_FINGER_MOTOR_ADJUST_PICKPLACE_INF,
    INSPIRE_PER_FINGER_MOTOR_ADJUST_PICKPLACE_REL_INF,
    INSPIRE_PER_FINGER_MOTOR_ADJUST_REL_TOOL_INF,
)
from dexumi.data_recording import VideoRecorder
from dexumi.data_recording.data_buffer import PoseInterpolator
from dexumi.data_recording.numeric_recorder import NumericRecorder
from dexumi.data_recording.record_manager import RecorderManager
from dexumi.encoder.fsr import FSRSensor
from dexumi.encoder.numeric import JointFrame
from dexumi.real_env.common.camera import CameraClient
from dexumi.real_env.common.dexhand import DexClient

# from dexumi.real_env.common.policy import PolicyClient
from dexumi.real_env.common.ur5 import UR5eClient
from dexumi.real_env.spacemouse import Spacemouse

# 定义手部到末端执行器的变换矩阵偏移量
x_offset = -0.0395
y_offset = -0.1342
z_offset = 0.0428
# T_ET: 手部到末端执行器的变换矩阵
T_ET = np.array(
    [
        [0, -1, 0, x_offset],
        [-1, 0, 0, y_offset],
        [0, 0, -1, z_offset],
        [0, 0, 0, 1],
    ]
)

# 观察历史长度（用于FSR传感器数据）
obs_horizon = 1
# FSR传感器的二值化阈值
binary_cutoff = [4000, 600]
# 手部初始位置（6个手指的初始角度）
initial_hand_pos = (
    np.array(
        [
            0.509,
            0.73700005,
            0.869,
            0.85700005,
            0.88000005,
            0.97300005,
        ]
    )
    / INSPIRE_HAND_MOTOR_SCALE_FACTOR
)


@click.command()
@click.option("-ms", "--max_pos_speed", type=float, default=0.25, help="最大位置速度")
@click.option("-mr", "--max_rot_speed", type=float, default=0.6, help="最大旋转速度")
@click.option("-f", "--frequency", type=float, default=10, help="控制频率")
@click.option(
    "-rc", "--enable_record_camera", is_flag=True, help="启用录制摄像头"
)
@click.option(
    "-mp",
    "--model_path",
    help="模型路径",
)
@click.option("-ckpt", "--ckpt", type=int, default=600, help="检查点编号")
@click.option(
    "-cl", "--camera_latency", type=float, default=0.185, help="摄像头延迟"
)
@click.option(
    "-hal", "--hand_action_latency", type=float, default=0.3, help="手部动作延迟"
)
@click.option(
    "-ral",
    "--robot_action_latency",
    type=float,
    default=0.170,
    help="机器人动作延迟",
)
@click.option("-eh", "--exec_horizon", type=int, default=8, help="执行视野")
@click.option(
    "-vp",
    "--video_record_path",
    type=str,
    default="video_record",
    help="视频录制保存路径",
)
@click.option(
    "-mep",
    "--match_episode_path",
    type=str,
    default=None,
    help="匹配episode文件夹路径",
)
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
):
    """
    主函数：实现基于策略的机器人手部控制系统
    
    该系统包含两个主要模式：
    1. 手动控制模式：使用SpaceMouse控制机器人运动
    2. 策略控制模式：使用训练好的策略模型自动控制机器人手部
    """
    # 初始化机器人客户端（UR5e机械臂）
    robot_client = UR5eClient()
    
    # 初始化手部客户端（Inspire Hand灵巧手）
    dexhand_client = DexClient(
        pub_address="ipc:///tmp/dex_stream",
        req_address="ipc:///tmp/dex_req",
        topic="dexhand",
    )
    
    # 初始化FSR力传感器
    fsr_sensor = FSRSensor("inspire_fsr", verbose=False, uart_port="/dev/ttyACM2")
    # fsr_sensor.start_streaming()
    
    # 获取所有OAK摄像头并初始化
    all_cameras = get_all_oak_cameras()
    obs_camera = OakCamera("obs camera", device_id=all_cameras[2])  # 观察摄像头
    camera_sources = [obs_camera]
    
    # 如果启用录制摄像头，添加录制摄像头
    if enable_record_camera:
        # record_camera = IphoneCamera(camera_name="record camera")
        record_camera = OakCamera("record camera", device_id=all_cameras[0])
        camera_sources.append(record_camera)
    
    # 初始化视频录制器
    video_recorder = VideoRecorder(
        record_fps=45,  # 录制帧率
        stream_fps=60,  # 流帧率
        video_record_path=video_record_path,
        camera_sources=camera_sources,
        frame_data_class=FrameData,
        verbose=False,
    )
    
    # 初始化数值录制器（用于FSR传感器数据）
    numeric_recorder = NumericRecorder(
        record_fps=45,
        stream_fps=60,
        record_path=video_record_path,
        numeric_sources=[fsr_sensor],
        frame_data_class=JointFrame,
        verbose=False,
    )
    
    # 初始化录制管理器
    recorder_manager = RecorderManager(
        recorders=[video_recorder, numeric_recorder], verbose=False
    )
    recorder_manager.start_streaming()

    dt = 1 / frequency  # 控制周期
    match_episode_folder = match_episode_path
    
    # 修复Python版本兼容性问题：将with语句改为传统格式
    with Spacemouse() as sm:
        # ==================== 手动控制模式 ====================
        while True:
            print("Ready!")
            
            # 如果提供了匹配episode路径，提取匹配帧
            if match_episode_folder is not None:
                print(
                    f"Extracting frames from match episode {recorder_manager.episode_id}"
                )
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
                
            command_latency = dt / 2  # 命令延迟
            
            # 获取机器人当前状态
            state = robot_client.get_state()
            target_pose = state.state["TargetTCPPose"]  # 目标TCP位姿
            recived_time = state.receive_time
            t_start = time.monotonic()
            print(time.time() - recived_time)

            iter_idx = 0
            # 手动控制循环
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample)  # 精确等待
                record_frame = recorder_manager.get_latest_frames()
                
                # 根据是否启用录制摄像头处理帧
                if enable_record_camera:
                    video_frame = record_frame["record camera"][-1]
                    viz_frame = video_frame.rgb.copy()

                obs_frame = record_frame["obs camera"][-1]
                
                # 叠加匹配初始帧和可视化帧
                if match_initial_frame is not None and enable_record_camera:
                    alpha = 0.5  # 透明度因子
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
                elif enable_record_camera:
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
                    
                # 显示观察帧并处理键盘输入
                if obs_frame is not None:
                    cv2.imshow("RGB", obs_frame.rgb)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):  # 按q退出手动控制模式
                        break
                    elif key == ord("x"):  # 按x退出程序
                        exit()
                        
                # 获取SpaceMouse状态并转换为机器人运动
                sm_state = sm.get_motion_state_transformed()
                print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)  # 位置增量
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)  # 旋转增量

                drot = st.Rotation.from_euler("xyz", drot_xyz)
                target_pose[:3] += dpos  # 更新目标位置
                target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()  # 更新目标旋转
                dpos = 0
                
                # 调度机器人路径点
                robot_client.schedule_waypoint(
                    target_pose, t_command_target - time.monotonic() + time.time()
                )
                precise_wait(t_cycle_end)
                iter_idx += 1

            cv2.destroyAllWindows()

            # ==================== 数据收集和处理 ====================
            # 接收机器人数据
            robot_frames = robot_client.receive_data()
            robot_timestamp = []
            robot_homogeneous_matrix = []
            for rf in robot_frames:
                robot_timestamp.append(rf.receive_time)
                robot_homogeneous_matrix.append(
                    vec6dof_to_homogeneous_matrix(
                        rf.state["ActualTCPPose"][:3],
                        rf.state["ActualTCPPose"][3:],
                    )
                )
            robot_timestamp = np.array(robot_timestamp)
            robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)
            t_now = time.time()
            print(
                robot_timestamp.min(),
                robot_timestamp.max(),
                t_now,
                t_now - robot_timestamp.max(),
            )

            # 初始化FSR观察数据
            fsr_obs = []
            fsr_obs = deque(maxlen=obs_horizon)
            fsr_value = np.array(fsr_sensor.get_numeric_frame().fsr_values)
            for _ in range(obs_horizon):
                fsr_obs.append(np.zeros(2))

            # ==================== 手部重置 ====================
            print(
                "resetting hand----------------------------------------------------------------------------------"
            )
            for i in range(3):
                dexhand_client.schedule_waypoint(
                    target_pos=initial_hand_pos,
                    target_time=time.time() + 0.05,
                )
                time.sleep(1)
            print(
                "reset done ----------------------------------------------------------------------------------"
            )
            # ==================== 策略控制模式 ====================
            from dexumi.real_env.real_policy import RealPolicy

            # 初始化策略模型
            policy = RealPolicy(
                model_path=model_path,
                ckpt=ckpt,
            )
            
            # 开始录制
            if recorder_manager.reset_episode_recording():
                click.echo("Starting recording...")
                recorder_manager.start_recording()
                
            inference_iter_time = exec_horizon * dt  # 推理迭代时间
            inference_fps = 1 / inference_iter_time  # 推理帧率
            print("inference_fps", inference_fps)
            # 策略控制主循环
            while True:
                with FrameRateContext(frame_rate=inference_fps):
                    # ==================== 观察数据收集 ====================
                    record_frame = recorder_manager.get_latest_frames()
                    
                    # 处理录制摄像头帧
                    if enable_record_camera:
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

                    # 处理观察摄像头帧
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
                    
                    # 如果启用FSR传感器，在图像上显示FSR值
                    if policy.model_cfg.dataset.enable_fsr:
                        # 在可视化帧上绘制FSR值
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
                        # 绘制二值化阈值
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR1 Binary: {int(fsr_value[0] < binary_cutoff[0])}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR2 Binary: {int(fsr_value[1] < binary_cutoff[1])}",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        
                    # 显示图像
                    cv2.imshow("obs frame", obs_frame_rgb)
                    if enable_record_camera:
                        cv2.imshow("record frame", viz_frame)
                        
                    # 处理键盘输入
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        if recorder_manager.stop_recording():
                            recorder_manager.save_recordings()
                        cv2.destroyAllWindows()
                        break
                    # ==================== FSR数据处理 ====================
                    if policy.model_cfg.dataset.enable_fsr:
                        print("Using FSR")
                        fsr_value = np.array(record_frame["inspire_fsr"][-1].fsr_values)
                        fsr_value = fsr_value.astype(np.float32)
                        # 应用二值化阈值 - 低于阈值的值变为1，高于阈值的值变为0
                        fsr_value_binary = (fsr_value < binary_cutoff).astype(
                            np.float32
                        )
                        fsr_obs.append(fsr_value_binary)
                        
                    # ==================== 策略推理 ====================
                    t_inference = time.monotonic()
                    # 摄像头延迟 + 传输时间
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
                    
                    # 使用策略模型预测动作
                    action = policy.predict_action(
                        None,
                        np.array(list(fsr_obs)).astype(np.float32)
                        if policy.model_cfg.dataset.enable_fsr
                        else None,
                        obs_frame.rgb[None, ...],
                    )
                    # ==================== 动作处理 ====================
                    # 转换为绝对动作
                    relative_pose = action[:, :6]  # 相对位姿（6DOF）
                    hand_action = action[:, 6:]    # 手部动作
                    
                    # 将相对位姿转换为齐次变换矩阵
                    relative_pose = np.array(
                        [
                            vec6dof_to_homogeneous_matrix(rp[:3], rp[3:])
                            for rp in relative_pose
                        ]
                    )
                    
                    # 处理手部动作（相对或绝对）
                    if policy.model_cfg.dataset.relative_hand_action:
                        print("Using relative hand action")
                        motor_current = dexhand_client.get_pos()
                        print("motor_current", motor_current)
                        motor_current = np.array(motor_current).reshape(1, -1)
                        print(
                            "revaltive hand_action",
                            np.round(hand_action * 1 / INSPIRE_HAND_MOTOR_SCALE_FACTOR),
                        )
                        hand_action = (
                            motor_current
                            + hand_action * 1 / INSPIRE_HAND_MOTOR_SCALE_FACTOR
                        )
                        hand_action += np.array(
                            # INSPIRE_PER_FINGER_MOTOR_ADJUST_PICKPLACE_REL_INF
                            # INSPIRE_PER_FINGER_MOTOR_ADJUST_EGG_INF
                            INSPIRE_PER_FINGER_MOTOR_ADJUST_REL_TOOL_INF
                        )
                        hand_action = hand_action.astype(int)
                    else:
                        print("Not using relative hand action")
                        hand_action = hand_action * 1 / INSPIRE_HAND_MOTOR_SCALE_FACTOR
                        hand_action += np.array(
                            # INSPIRE_PER_FINGER_MOTOR_ADJUST_PICKPLACE_INF
                            # INSPIRE_PER_FINGER_MOTOR_ADJUST_PICKPLACE_REL_INF,
                            # INSPIRE_PER_FINGER_MOTOR_ADJUST_EGG_INF
                            INSPIRE_PER_FINGER_MOTOR_ADJUST_ABS_TOOL_INF
                        )
                        print("hand_action", hand_action.astype(int))
                    hand_action = np.clip(hand_action, 0, 1000)  # 限制手部动作范围

                    # ==================== 机器人位姿对齐 ====================
                    # 获取图像捕获时的机器人位姿
                    robot_frames = robot_client.get_state_history()
                    robot_timestamp = []
                    robot_homogeneous_matrix = []
                    for rf in robot_frames:
                        robot_timestamp.append(rf.receive_time)
                        robot_homogeneous_matrix.append(
                            vec6dof_to_homogeneous_matrix(
                                rf.state["ActualTCPPose"][:3],
                                rf.state["ActualTCPPose"][3:],
                            )
                        )
                    robot_timestamp = np.array(robot_timestamp)
                    robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)
                    
                    # 创建位姿插值器
                    robot_pose_interpolator = PoseInterpolator(
                        timestamps=robot_timestamp,
                        homogeneous_matrix=robot_homogeneous_matrix,
                    )
                    aligned_pose = robot_pose_interpolator([t_actual_inference])[0]
                    ee_aligned_pose = homogeneous_matrix_to_6dof(aligned_pose)
                    
                    # 构建变换矩阵
                    T_BE = np.eye(4)
                    T_BE[:3, :3] = st.Rotation.from_rotvec(
                        ee_aligned_pose[3:]
                    ).as_matrix()
                    T_BE[:3, -1] = ee_aligned_pose[:3]
                    T_BN = np.zeros_like(relative_pose)
                    
                    # 计算目标位姿
                    for iter_idx in range(len(relative_pose)):
                        T_BN[iter_idx] = (
                            T_BE
                            @ T_ET
                            @ relative_pose[iter_idx]
                            @ invert_transformation(T_ET)
                        )
                        
                    # ==================== 动作调度 ====================
                    # 丢弃过去的动作
                    n_action = T_BN.shape[0]
                    t_exec = time.monotonic()
                    robot_scheduled = 0
                    hand_scheduled = 0

                    # 处理机器人路径点
                    robot_times = t_actual_inference + np.arange(n_action) * dt
                    valid_robot_idx = robot_times >= t_exec + robot_action_latency + dt
                    # 转换为全局时间
                    robot_times = robot_times - time.monotonic() + time.time()
                    for k in np.where(valid_robot_idx)[0]:
                        target_pose = np.zeros(6)
                        target_pose[:3] = T_BN[k, :3, -1]
                        target_pose[3:] = st.Rotation.from_matrix(
                            T_BN[k, :3, :3]
                        ).as_rotvec()
                        robot_client.schedule_waypoint(target_pose, robot_times[k])
                        robot_scheduled += 1

                    # 处理手部路径点
                    hand_times = t_actual_inference + np.arange(n_action) * dt
                    valid_hand_idx = hand_times >= t_exec + hand_action_latency + dt
                    # 转换为全局时间
                    hand_times = hand_times - time.monotonic() + time.time()
                    for k in np.where(valid_hand_idx)[0]:
                        target_hand_action = hand_action[k].astype(int)
                        dexhand_client.schedule_waypoint(
                            target_hand_action, hand_times[k]
                        )
                        hand_scheduled += 1

                    print(
                        f"Scheduled actions: {robot_scheduled} robot waypoints, {hand_scheduled} hand waypoints"
                    )


if __name__ == "__main__":
    main()
