# -*- coding: utf-8 -*-
"""
XHand机械手策略评估脚本
用于实时控制UR5e机器人和XHand灵巧手，结合视觉和触觉反馈执行学习到的策略
支持手动控制和自动策略执行两种模式
"""

import os  # 文件路径操作
import time  # 时间相关功能，用于延时和时间戳
from collections import deque  # 双端队列，用于存储观测历史

import click  # 命令行接口框架，用于解析命令行参数
import cv2  # OpenCV计算机视觉库，用于图像处理和显示
import numpy as np  # 数值计算库，用于矩阵运算
import scipy.spatial.transform as st  # 空间变换库，用于旋转矩阵计算

# DexUMI框架相关导入
from dexumi.camera.camera import FrameData  # 相机帧数据结构
from dexumi.camera.iphone_camera import IphoneCamera  # iPhone相机接口
from dexumi.camera.oak_camera import OakCamera, get_all_oak_cameras  # OAK相机接口和设备发现
from dexumi.common.frame_manager import FrameRateContext  # 帧率控制上下文管理器
from dexumi.common.precise_sleep import precise_wait  # 精确等待函数，用于时序控制
from dexumi.common.utility.matrix import (  # 矩阵变换工具函数
    convert_homogeneous_matrix,  # 齐次矩阵转换
    homogeneous_matrix_to_6dof,  # 齐次矩阵转6DOF姿态
    invert_transformation,  # 变换矩阵求逆
    relative_transformation,  # 相对变换计算
    vec6dof_to_homogeneous_matrix,  # 6DOF向量转齐次矩阵
    visualize_multiple_frames_and_points,  # 多坐标系可视化
)
from dexumi.common.utility.video import (  # 视频处理工具
    extract_frames_videos,  # 从视频提取帧
)
from dexumi.constants import (  # 系统常量
    XHAND_HAND_MOTOR_SCALE_FACTOR,  # XHand电机缩放因子
)
from dexumi.data_recording import VideoRecorder  # 视频录制器
from dexumi.data_recording.data_buffer import PoseInterpolator  # 姿态插值器
from dexumi.data_recording.numeric_recorder import NumericRecorder  # 数值数据录制器
from dexumi.data_recording.record_manager import RecorderManager  # 录制管理器
from dexumi.real_env.common.dexhand import DexClient  # 灵巧手客户端

# from dexumi.real_env.common.policy import PolicyClient  # 策略客户端（已注释）
from dexumi.real_env.common.ur5 import UR5eClient  # UR5e机器人客户端
from dexumi.real_env.spacemouse import Spacemouse  # 3D鼠标接口


def compute_total_force_per_finger(all_fsr_observations):
    """
    计算每个手指的总力值

    参数:
    all_fsr_observations (numpy.ndarray): 形状为(m, n, 3)的数组，其中:
        - m 是观测次数
        - n 是手指数量  
        - 3 是xyz力分量

    返回:
    numpy.ndarray: 形状为(m, n)的数组，包含每个手指的总力幅值
    """
    # 计算3D力向量的欧几里得范数（幅值）
    # 计算每个手指在每次观测中的sqrt(x² + y² + z²)
    total_force = np.linalg.norm(all_fsr_observations, axis=2)

    return total_force


# 末端执行器到工具坐标系的变换参数（单位：米）
x_offset = -0.0395  # X轴偏移量
y_offset = -0.1342  # Y轴偏移量  
z_offset = 0.0428   # Z轴偏移量

# 末端执行器到工具坐标系的齐次变换矩阵 (End-effector to Tool)
# 这个矩阵定义了机器人末端执行器坐标系到工具（手爪）坐标系的变换关系
T_ET = np.array(
    [
        [0, -1, 0, x_offset],  # 第一行：X轴变换
        [-1, 0, 0, y_offset],  # 第二行：Y轴变换
        [0, 0, -1, z_offset],  # 第三行：Z轴变换
        [0, 0, 0, 1],          # 第四行：齐次坐标
    ]
)

# 观测历史长度，用于时序数据处理
obs_horizon = 1

# FSR传感器二值化阈值，用于判断是否有接触
# 对应三个手指的力传感器阈值
binary_cutoff = [10, 10, 10]

# 灵巧手初始位置配置（12个电机的角度值，单位：弧度）
# 这个配置定义了手指的初始张开状态
initial_hand_pos = np.array(
    [
        0.92755819,  # 拇指关节1
        0.52026953,  # 拇指关节2
        0.22831853,  # 拇指关节3
        0.0707963,   # 拇指关节4
        1.1,         # 食指关节1
        0.15707963,  # 食指关节2
        0.95,        # 食指关节3
        0.12217305,  # 食指关节4
        1.0392188,   # 中指关节1
        0.03490659,  # 中指关节2
        1.0078164,   # 中指关节3
        0.17453293,  # 中指关节4
    ]
)


@click.command()
# 命令行参数定义
@click.option("-ms", "--max_pos_speed", type=float, default=0.1, help="最大位置速度 (m/s)")
@click.option("-mr", "--max_rot_speed", type=float, default=0.6, help="最大旋转速度 (rad/s)")
@click.option("-f", "--frequency", type=float, default=10, help="控制频率 (Hz)")
@click.option(
    "-rc", "--enable_record_camera", is_flag=True, help="启用录制相机"
)
@click.option(
    "-mp",
    "--model_path",
    help="模型文件路径",
)
@click.option("-ckpt", "--ckpt", type=int, default=600, help="检查点编号")
@click.option(
    "-cl", "--camera_latency", type=float, default=0.185, help="相机延迟 (秒)"
)
@click.option(
    "-hal", "--hand_action_latency", type=float, default=0.3, help="手部动作延迟 (秒)"
)
@click.option(
    "-ral",
    "--robot_action_latency",
    type=float,
    default=0.170,
    help="机器人动作延迟 (秒)",
)
@click.option("-eh", "--exec_horizon", type=int, default=8, help="执行时间窗口")
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
    frequency,           # 控制频率
    max_pos_speed,      # 最大位置速度
    max_rot_speed,      # 最大旋转速度
    enable_record_camera, # 是否启用录制相机
    model_path,         # 模型路径
    ckpt,              # 检查点
    camera_latency,     # 相机延迟
    hand_action_latency, # 手部动作延迟
    robot_action_latency, # 机器人动作延迟
    exec_horizon,       # 执行时间窗口
    video_record_path,  # 视频录制路径
    match_episode_path, # 匹配episode路径
):
    # 初始化硬件客户端
    robot_client = UR5eClient()  # 创建UR5e机器人客户端
    dexhand_client = DexClient(  # 创建灵巧手客户端
        pub_address="ipc:///tmp/dex_stream",  # 发布地址
        req_address="ipc:///tmp/dex_req",     # 请求地址
        topic="dexhand",                     # 话题名称
    )
    
    # 相机系统初始化
    all_cameras = get_all_oak_cameras()  # 获取所有可用的OAK相机
    obs_camera = OakCamera("obs camera", device_id=all_cameras[1])  # 观测相机（用于策略输入）
    camera_sources = [obs_camera]  # 相机源列表
    
    if enable_record_camera:  # 如果启用录制相机
        # record_camera = IphoneCamera(camera_name="record camera")  # iPhone相机选项（已注释）
        record_camera = OakCamera("record camera", device_id=all_cameras[0])  # 录制相机
        camera_sources.append(record_camera)  # 添加到相机源列表
    
    # 视频录制系统初始化
    video_recorder = VideoRecorder(
        record_fps=45,                    # 录制帧率
        stream_fps=60,                    # 流媒体帧率
        video_record_path=video_record_path,  # 录制保存路径
        camera_sources=camera_sources,    # 相机源
        frame_data_class=FrameData,      # 帧数据类
        verbose=False,                   # 不显示详细信息
    )
    recorder_manager = RecorderManager(  # 录制管理器
        recorders=[video_recorder],      # 录制器列表
        verbose=False,                   # 不显示详细信息
    )
    recorder_manager.start_streaming()   # 开始流媒体

    # 计算控制周期时间
    dt = 1 / frequency  # 控制周期（秒）
    match_episode_folder = match_episode_path  # 匹配episode文件夹路径
    
    # 修复Python版本兼容性问题：移除多余的括号
    with Spacemouse() as sm:  # 初始化3D鼠标
        while True:  # 主控制循环
            print("Ready!")  # 准备就绪提示
            
            # 处理匹配episode（用于对比显示）
            if match_episode_folder is not None:
                print(
                    f"Extracting frames from match episode {recorder_manager.episode_id}"
                )
                # 从匹配的episode视频中提取帧
                match_episode = extract_frames_videos(
                    os.path.join(
                        match_episode_folder,
                        f"episode_{recorder_manager.episode_id}/camera_1.mp4",
                    ),
                    BGR2RGB=True,  # 转换为RGB格式
                )
                match_initial_frame = match_episode[0]  # 获取初始帧用于对比
            else:
                print("No match episode folder provided")  # 没有提供匹配episode文件夹
                match_initial_frame = None
            
            # 初始化控制参数
            command_latency = dt / 2  # 命令延迟（控制周期的一半）
            state = robot_client.get_state()  # 获取机器人当前状态
            target_pose = state.state["TargetTCPPose"]  # 目标TCP姿态
            recived_time = state.receive_time  # 接收时间
            t_start = time.monotonic()  # 记录开始时间（单调时间）
            print(time.time() - recived_time)  # 打印时间差

            iter_idx = 0  # 迭代索引
            
            # 手动控制内循环
            while True:
                # 计算时序控制点
                t_cycle_end = t_start + (iter_idx + 1) * dt  # 周期结束时间
                t_sample = t_cycle_end - command_latency      # 采样时间点
                t_command_target = t_cycle_end + dt           # 命令目标时间

                precise_wait(t_sample)  # 精确等待到采样时间
                record_frame = recorder_manager.get_latest_frames()  # 获取最新帧
                
                # 根据录制相机标志处理帧
                if enable_record_camera:
                    video_frame = record_frame["record camera"][-1]  # 获取录制相机帧
                    viz_frame = video_frame.rgb.copy()  # 复制用于可视化

                obs_frame = record_frame["obs camera"][-1]  # 获取观测相机帧
                
                # 叠加匹配初始帧和可视化帧（用于对比显示）
                if match_initial_frame is not None and enable_record_camera:
                    alpha = 0.5  # 透明度因子
                    overlay = cv2.addWeighted(
                        match_initial_frame, alpha, viz_frame, 1 - alpha, 0
                    )  # 加权叠加两个图像
                    cv2.putText(  # 在叠加图像上添加文本
                        overlay,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # 绿色文字
                        2,
                    )
                    cv2.imshow("Overlay", overlay)  # 显示叠加图像
                elif enable_record_camera:
                    cv2.putText(  # 在可视化帧上添加文本
                        viz_frame,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # 绿色文字
                        2,
                    )
                    cv2.imshow("VIZ", viz_frame)  # 显示可视化帧
                    
                # 显示观测帧
                if obs_frame is not None:
                    cv2.imshow("RGB", obs_frame.rgb)  # 显示RGB图像
                    key = cv2.waitKey(1) & 0xFF  # 检测键盘输入
                    if key == ord("q"):  # 按'q'退出内循环
                        break
                    elif key == ord("x"):  # 按'x'退出程序
                        exit()
                
                # 3D鼠标控制处理
                sm_state = sm.get_motion_state_transformed()  # 获取3D鼠标变换后的运动状态
                dpos = sm_state[:3] * (max_pos_speed / frequency)     # 位置增量
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency) # 旋转增量（欧拉角）

                # 应用旋转和位置变换
                drot = st.Rotation.from_euler("xyz", drot_xyz)  # 从欧拉角创建旋转对象
                target_pose[:3] += dpos  # 更新位置
                target_pose[3:] = (  # 更新旋转（旋转向量表示）
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()
                dpos = 0  # 重置位置增量
                
                # 向机器人发送路径点命令
                robot_client.schedule_waypoint(
                    target_pose, t_command_target - time.monotonic() + time.time()
                )
                precise_wait(t_cycle_end)  # 精确等待到周期结束
                iter_idx += 1  # 增加迭代索引

            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

            # ========== 机器人状态数据处理 ==========
            robot_frames = robot_client.receive_data()  # 接收机器人数据
            robot_timestamp = []      # 机器人时间戳列表
            robot_homogeneous_matrix = []  # 机器人齐次变换矩阵列表
            
            # 处理每一帧机器人数据
            for rf in robot_frames:
                robot_timestamp.append(rf.receive_time)  # 添加接收时间
                robot_homogeneous_matrix.append(  # 将6DOF姿态转换为齐次矩阵
                    vec6dof_to_homogeneous_matrix(
                        rf.state["ActualTCPPose"][:3],  # 位置部分
                        rf.state["ActualTCPPose"][3:],  # 旋转部分
                    )
                )
            robot_timestamp = np.array(robot_timestamp)  # 转换为numpy数组
            robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)  # 转换为numpy数组
            
            t_now = time.time()  # 当前时间
            print(  # 打印时间信息用于调试
                robot_timestamp.min(),      # 最早时间戳
                robot_timestamp.max(),      # 最晚时间戳  
                t_now,                     # 当前时间
                t_now - robot_timestamp.max(),  # 延迟时间
            )

            # ========== FSR触觉传感器初始化 ==========
            fsr_obs = []  # FSR观测列表（将被下面的deque覆盖）
            fsr_obs = deque(maxlen=obs_horizon)  # FSR观测历史队列，最大长度为观测时间窗口
            fsr_raw_obs = dexhand_client.get_tactile(calc=True)  # 获取触觉传感器原始数据
            
            # 重新整形FSR原始观测数据以添加批次维度
            fsr_raw_obs = fsr_raw_obs[None, ...]  # 在开始位置添加维度 [1, n_fingers, 3]
            fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]  # 计算每个手指的总力值
            fsr_value = np.array(fsr_raw_obs[:3])  # 取前三个手指的力值
            print("fsr_value", fsr_value)  # 打印FSR值用于调试
            
            # 初始化FSR观测历史（填充零值）
            for _ in range(obs_horizon):
                fsr_obs.append(np.zeros(2))  # 添加零向量到观测历史

            print(
                "resetting hand----------------------------------------------------------------------------------"
            )

            # ========== 灵巧手复位 ==========
            virtual_hand_pos = initial_hand_pos  # 虚拟手部位置设为初始位置
            for i in range(3):  # 重复3次确保复位成功
                dexhand_client.schedule_waypoint(  # 调度路径点
                    target_pos=initial_hand_pos,    # 目标位置为初始位置
                    target_time=time.time() + 0.05, # 目标时间为当前时间+50ms
                )
                time.sleep(1)  # 等待1秒
            print(
                "reset done ----------------------------------------------------------------------------------"
            )
            # ========== 策略模型加载 ==========
            from dexumi.real_env.real_policy import RealPolicy  # 导入实时策略类

            policy = RealPolicy(  # 创建策略实例
                model_path=model_path,  # 模型文件路径
                ckpt=ckpt,             # 检查点编号
            )
            
            # ========== 开始录制 ==========
            if recorder_manager.reset_episode_recording():  # 重置episode录制
                click.echo("Starting recording...")  # 提示开始录制
                recorder_manager.start_recording()   # 开始录制

            # ========== 策略执行参数计算 ==========
            inference_iter_time = exec_horizon * dt  # 推理迭代时间 = 执行时间窗口 × 控制周期
            inference_fps = 1 / inference_iter_time  # 推理帧率 = 1 / 推理迭代时间
            print("inference_fps", inference_fps)    # 打印推理帧率
            
            # ========== 策略执行主循环 ==========
            while True:
                with FrameRateContext(frame_rate=inference_fps):  # 帧率控制上下文
                    # ========== 收集观测数据 ==========
                    record_frame = recorder_manager.get_latest_frames()  # 获取最新帧
                    
                    # 处理录制相机帧（如果启用）
                    if enable_record_camera:
                        video_frame = record_frame["record camera"][-1]  # 获取录制相机最新帧
                        viz_frame = video_frame.rgb.copy()  # 复制RGB数据用于可视化
                        cv2.putText(  # 在可视化帧上添加episode信息
                            viz_frame,
                            f"Episode: {recorder_manager.episode_id}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )

                    # 处理观测相机帧（用于策略输入）
                    obs_frame = record_frame["obs camera"][-1]  # 获取观测相机最新帧
                    obs_frame_recieved_time = obs_frame.receive_time  # 记录帧接收时间（用于延迟补偿）
                    obs_frame_rgb = obs_frame.rgb.copy()  # 复制RGB数据用于显示
                    cv2.putText(  # 在观测帧上添加episode信息
                        obs_frame_rgb,
                        f"Episode: {recorder_manager.episode_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # 绿色文字
                        2,
                    )
                    # ========== FSR传感器数据显示 ==========
                    if policy.model_cfg.dataset.enable_fsr:  # 如果策略模型启用FSR传感器
                        # 在观测帧上绘制FSR原始值
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR1: {fsr_value[0]:.0f}",  # 第一个手指的力值
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR2: {fsr_value[1]:.0f}",  # 第二个手指的力值
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )
                        
                        # 绘制二值化阈值判断结果
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR1 Binary: {int(fsr_value[0] > binary_cutoff[0])}",  # 第一个手指是否超过阈值
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR2 Binary: {int(fsr_value[1] > binary_cutoff[1])}",  # 第二个手指是否超过阈值
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )
                        cv2.putText(
                            obs_frame_rgb,
                            f"FSR3 Binary: {int(fsr_value[2] > binary_cutoff[2])}",  # 第三个手指是否超过阈值
                            (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),  # 绿色文字
                            2,
                        )
                    # ========== 显示图像和键盘控制 ==========
                    cv2.imshow("obs frame", obs_frame_rgb)  # 显示观测帧
                    if enable_record_camera:
                        cv2.imshow("record frame", viz_frame)  # 显示录制帧
                    
                    key = cv2.waitKey(1) & 0xFF  # 检测键盘输入
                    if key == ord("q"):  # 按'q'键：保存录制并退出
                        if recorder_manager.stop_recording():
                            recorder_manager.save_recordings()
                        cv2.destroyAllWindows()
                        break
                    elif key == ord("a"):  # 按'a'键：放弃录制并退出
                        if recorder_manager.stop_recording():
                            recorder_manager.clear_recording()
                        break
                    
                    # ========== 更新FSR传感器数据 ==========
                    if policy.model_cfg.dataset.enable_fsr:  # 如果策略使用FSR传感器
                        print("Using FSR")  # 调试信息
                        fsr_raw_obs = dexhand_client.get_tactile(calc=True)  # 获取最新触觉数据
                        print("raw", fsr_raw_obs)  # 打印原始数据
                        
                        # 重新整形FSR原始观测数据以添加批次维度
                        fsr_raw_obs = fsr_raw_obs[
                            None, ...
                        ]  # 在开始位置添加维度 [1, n_fingers, 3]
                        fsr_raw_obs = compute_total_force_per_finger(fsr_raw_obs)[0]  # 计算总力值
                        fsr_value = np.array(fsr_raw_obs[:3])  # 取前三个手指
                        print("fsr_value", fsr_value)  # 打印处理后的FSR值
                        fsr_value = fsr_value.astype(np.float32)  # 转换为float32类型
                        
                        # 应用二值化阈值
                        fsr_value_binary = (fsr_value >= binary_cutoff).astype(
                            np.float32
                        )  # 大于等于阈值设为1，否则为0
                        fsr_obs.append(fsr_value_binary)  # 添加到观测历史队列
                    
                    # ========== 策略推理时间计算 ==========
                    t_inference = time.monotonic()  # 推理开始时间（单调时间）
                    
                    # 计算相机延迟 + 传输时间
                    print(
                        "t_inference|obs_frame_recieved_time",
                        t_inference,
                        obs_frame_recieved_time,
                    )
                    camera_total_latency = (  # 总相机延迟 = 相机延迟 + 传输延迟
                        camera_latency + t_inference - obs_frame_recieved_time
                    )
                    print("camera_total_latency", camera_total_latency)
                    t_actual_inference = t_inference - camera_total_latency  # 实际推理时间（补偿延迟）
                    # ========== 策略动作预测 ==========
                    action = policy.predict_action(
                        None,  # 不使用状态输入
                        np.array(list(fsr_obs)).astype(np.float32)  # FSR传感器历史数据
                        if policy.model_cfg.dataset.enable_fsr
                        else None,  # 如果不启用FSR则传入None
                        obs_frame.rgb[None, ...],  # 观测图像（添加batch维度）
                    )
                    
                    # ========== 动作分解和转换 ==========
                    # 将动作分解为机器人姿态和手部动作
                    relative_pose = action[:, :6]  # 前6维：相对姿态（位置+旋转） 
                    hand_action = action[:, 6:]    # 后面维度：手部动作（关节角度）
                    
                    # 将相对姿态从6DOF向量转换为齐次变换矩阵
                    relative_pose = np.array(
                        [
                            vec6dof_to_homogeneous_matrix(rp[:3], rp[3:])  # 位置+旋转 -> 齐次矩阵
                            for rp in relative_pose
                        ]
                    )
                    # ========== 手部动作处理 ==========
                    if policy.model_cfg.dataset.relative_hand_action:  # 如果使用相对手部动作
                        print("Using relative hand action")  # 调试信息
                        # motor_current = dexhand_client.get_pos()  # 获取当前电机位置（已注释）
                        motor_current = virtual_hand_pos  # 使用虚拟手部位置
                        # print("motor_current", motor_current)  # 打印当前电机位置（已注释）
                        motor_current = np.array(motor_current).reshape(1, -1)  # 重塑为行向量
                        print(  # 打印相对手部动作（调试用）
                            "revaltive hand_action",
                            np.round(
                                hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR, 2  # 缩放并四舍五入
                            ),
                        )
                        # 计算绝对手部动作 = 当前位置 + 相对动作
                        hand_action = (
                            motor_current
                            + hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR  # 应用缩放因子
                        )
                        offset = np.array([0.0] * 12)  # 偏移量（全零）
                        hand_action += offset  # 应用偏移量
                    else:  # 如果使用绝对手部动作
                        print("Not using relative hand action")  # 调试信息
                        hand_action = hand_action * 1 / XHAND_HAND_MOTOR_SCALE_FACTOR  # 直接应用缩放因子
                        offset = np.array([0.0] * 12)  # 创建偏移量数组
                        offset[0] = 0.025  # 设置第一个电机的偏移量

                        hand_action += offset  # 应用偏移量

                    # ========== 机器人姿态插值和坐标系变换 ==========
                    # 获取图像捕获时的机器人姿态
                    robot_frames = robot_client.get_state_history()  # 获取机器人状态历史
                    robot_timestamp = []  # 时间戳列表
                    robot_homogeneous_matrix = []  # 齐次变换矩阵列表
                    
                    # 处理机器人历史数据
                    for rf in robot_frames:
                        robot_timestamp.append(rf.receive_time)  # 添加接收时间
                        robot_homogeneous_matrix.append(  # 转换为齐次矩阵
                            vec6dof_to_homogeneous_matrix(
                                rf.state["ActualTCPPose"][:3],  # 位置
                                rf.state["ActualTCPPose"][3:],  # 旋转
                            )
                        )
                    robot_timestamp = np.array(robot_timestamp)  # 转换为numpy数组
                    robot_homogeneous_matrix = np.array(robot_homogeneous_matrix)  # 转换为numpy数组
                    
                    # 创建姿态插值器
                    robot_pose_interpolator = PoseInterpolator(
                        timestamps=robot_timestamp,        # 时间戳
                        homogeneous_matrix=robot_homogeneous_matrix,  # 齐次矩阵
                    )
                    
                    # 插值得到推理时刻的机器人姿态
                    aligned_pose = robot_pose_interpolator([t_actual_inference])[0]  # 插值姿态
                    ee_aligned_pose = homogeneous_matrix_to_6dof(aligned_pose)  # 转换为6DOF
                    
                    # 构建基座到末端执行器的变换矩阵 T_BE
                    T_BE = np.eye(4)  # 4x4单位矩阵
                    T_BE[:3, :3] = st.Rotation.from_rotvec(  # 设置旋转部分
                        ee_aligned_pose[3:]  # 旋转向量
                    ).as_matrix()
                    T_BE[:3, -1] = ee_aligned_pose[:3]  # 设置位移部分
                    
                    # 计算基座坐标系下的目标姿态 T_BN
                    T_BN = np.zeros_like(relative_pose)  # 初始化结果数组
                    for iter_idx in range(len(relative_pose)):  # 遍历每个动作
                        # T_BN = T_BE @ T_ET @ T_EN @ T_ET^(-1)
                        # 其中T_EN是末端执行器坐标系下的相对变换
                        T_BN[iter_idx] = (
                            T_BE  # 基座到末端执行器
                            @ T_ET  # 末端执行器到工具
                            @ relative_pose[iter_idx]  # 工具坐标系下的相对变换
                            @ invert_transformation(T_ET)  # 工具到末端执行器
                        )
                    # ========== 动作调度和执行 ==========
                    # 丢弃过时的动作（在过去的时间点）
                    n_action = T_BN.shape[0]  # 动作数量
                    t_exec = time.monotonic()  # 当前执行时间
                    robot_scheduled = 0  # 已调度的机器人路径点计数
                    hand_scheduled = 0   # 已调度的手部路径点计数

                    # ========== 处理机器人路径点 ==========
                    robot_times = t_actual_inference + np.arange(n_action) * dt  # 计算每个动作的执行时间
                    # 筛选有效的机器人动作（考虑延迟）
                    valid_robot_idx = robot_times >= t_exec + robot_action_latency + dt
                    # 转换为全局时间（wall clock time）
                    robot_times = robot_times - time.monotonic() + time.time()
                    
                    # 调度有效的机器人路径点
                    for k in np.where(valid_robot_idx)[0]:  # 遍历有效索引
                        target_pose = np.zeros(6)  # 创建6DOF目标姿态
                        target_pose[:3] = T_BN[k, :3, -1]  # 提取位置部分
                        target_pose[3:] = st.Rotation.from_matrix(  # 提取旋转部分
                            T_BN[k, :3, :3]  # 旋转矩阵
                        ).as_rotvec()  # 转换为旋转向量
                        robot_client.schedule_waypoint(target_pose, robot_times[k])  # 调度路径点
                        robot_scheduled += 1  # 增加计数

                    # ========== 处理手部路径点 ==========
                    hand_times = t_actual_inference + np.arange(n_action) * dt  # 计算每个动作的执行时间
                    # 筛选有效的手部动作（考虑延迟）
                    valid_hand_idx = hand_times >= t_exec + hand_action_latency + dt
                    # 转换为全局时间（wall clock time）
                    hand_times = hand_times - time.monotonic() + time.time()
                    
                    # 调度有效的手部路径点
                    for k in np.where(valid_hand_idx)[0]:  # 遍历有效索引
                        target_hand_action = hand_action[k]  # 获取目标手部动作
                        dexhand_client.schedule_waypoint(  # 调度手部路径点
                            target_hand_action, hand_times[k]
                        )
                        hand_scheduled += 1  # 增加计数

                    # 打印调度统计信息
                    print(
                        f"Scheduled actions: {robot_scheduled} robot waypoints, {hand_scheduled} hand waypoints"
                    )
                    # 更新虚拟手部位置（用于下一次相对动作计算）
                    virtual_hand_pos = hand_action[exec_horizon + 1]


# ========== 程序入口点 ==========
if __name__ == "__main__":
    main()  # 运行主函数
