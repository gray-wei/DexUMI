"""
XHand + Franka 多模态数据采集脚本

核心特性:
1. 独立采集 - 相机和机器人数据独立采集，各自最优频率
2. 高精度时间戳 - 硬件时间戳 + 系统时间戳双重记录
3. 离线对齐 - 采集完成后通过独立脚本进行时间戳对齐
4. 无同步等待 - 消除复杂的在线同步逻辑
5. 数据兼容性 - 数据格式完全兼容现有的convert_pickle_to_zarr.py
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import time
import pickle
import os
import shutil
import requests
import argparse
from pynput import keyboard
from typing import Optional, Dict, List, Tuple
from queue import Queue, Empty
from collections import deque
import logging
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class CollectionState(Enum):
    """采集状态枚举"""
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2


class CameraCollector:
    """独立相机采集器 - 高帧率采集，记录硬件时间戳"""
    
    def __init__(self):
        self.camera_data = []  # 存储采集的相机帧
        self._state = CollectionState.STOPPED
        self._state_lock = threading.Lock()
        self.lock = threading.Lock()
        self._last_frame_hash = {}  # 用于检测重复帧
        
    def start_collecting(self):
        """开始新的采集会话（清空数据）"""
        with self._state_lock:
            self._state = CollectionState.RUNNING
        with self.lock:
            self.camera_data = []
            self._last_frame_hash = {}
        
    def resume_collecting(self, pipelines=None):
        """恢复采集（保持现有数据）"""
        if pipelines:
            self._clear_pipeline_cache(pipelines)
        with self._state_lock:
            self._state = CollectionState.RUNNING
            
    def pause_collecting(self):
        """暂停采集（保持数据和线程）"""
        with self._state_lock:
            self._state = CollectionState.PAUSED
        
    def stop_collecting(self):
        """停止采集相机数据"""
        with self._state_lock:
            self._state = CollectionState.STOPPED
            
    def get_state(self) -> CollectionState:
        """获取当前采集状态"""
        with self._state_lock:
            return self._state
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.get_state() == CollectionState.RUNNING
        
    def _clear_pipeline_cache(self, pipelines):
        """清理相机pipeline缓存"""
        # 丢弃几帧确保获取新数据
        for cam_info in pipelines:
            try:
                # 快速丢弃最多5帧缓存
                for _ in range(5):
                    frames = cam_info['pipeline'].poll_for_frames()
                    if not frames:
                        break
            except Exception as e:
                logger.debug(f"清理相机 {cam_info['camera_id']} 缓存失败: {e}")
                
    def _validate_frame(self, frame_data: Dict, cam_id: int) -> bool:
        """验证帧数据的有效性"""
        if not frame_data:
            return False
            
        rgb = frame_data.get('rgb')
        if rgb is None:
            return False
            
        # 检查时间戳是否合理（不能太旧）
        sys_timestamp = frame_data.get('system_timestamp', 0)
        current_time = time.time()
        if current_time - sys_timestamp > 0.5:  # 超过0.5秒认为是旧数据
            logger.debug(f"警告: 相机{cam_id}数据过旧, 时间差: {current_time - sys_timestamp:.3f}s")
            return False
            
        # 检查图像内容是否变化（可选 - 用于调试）
        current_hash = hash(rgb[::8, ::8].tobytes())  # 使用降采样避免计算开销
        if cam_id in self._last_frame_hash:
            if current_hash == self._last_frame_hash[cam_id]:
                logger.debug(f"警告: 相机{cam_id}图像内容可能重复")
                # 注意：这里不返回False，因为某些场景下图像确实可能相同
        self._last_frame_hash[cam_id] = current_hash
            
        return True
        
    def get_current_frame(self) -> Optional[Dict]:
        """获取最新的相机帧（用于实时显示）"""
        with self.lock:
            if self.camera_data:
                return self.camera_data[-1]
        return None
    
    def get_collected_data(self) -> List[Dict]:
        """获取已采集的所有相机数据"""
        with self.lock:
            return self.camera_data.copy()
    
    def clear_data(self):
        """清空采集数据"""
        with self.lock:
            self.camera_data = []


class RobotDataCollector:
    """独立机器人数据采集器 - 固定频率采集HTTP数据"""
    
    def __init__(self, url: str = "http://127.0.0.1:5000/"):
        self.url = url
        self.robot_data = []  # 存储采集的机器人数据
        self._state = CollectionState.STOPPED
        self._state_lock = threading.Lock()
        self.lock = threading.Lock()
        
    def start_collecting(self):
        """开始新的采集会话（清空数据）"""
        with self._state_lock:
            self._state = CollectionState.RUNNING
        with self.lock:
            self.robot_data = []
            
    def resume_collecting(self):
        """恢复采集（保持现有数据）"""
        with self._state_lock:
            self._state = CollectionState.RUNNING
            
    def pause_collecting(self):
        """暂停采集（保持数据和线程）"""
        with self._state_lock:
            self._state = CollectionState.PAUSED
        
    def stop_collecting(self):
        """停止采集机器人数据"""
        with self._state_lock:
            self._state = CollectionState.STOPPED
            
    def get_state(self) -> CollectionState:
        """获取当前采集状态"""
        with self._state_lock:
            return self._state
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.get_state() == CollectionState.RUNNING
        
    def get_current_state(self) -> Optional[Dict]:
        """获取最新的机器人状态（用于实时显示）"""
        with self.lock:
            if self.robot_data:
                return self.robot_data[-1]
        return None
    
    def get_collected_data(self) -> List[Dict]:
        """获取已采集的所有机器人数据"""
        with self.lock:
            return self.robot_data.copy()
    
    def clear_data(self):
        """清空采集数据"""
        with self.lock:
            self.robot_data = []


class CollectionPhase(Enum):
    """采集阶段枚举"""
    WAITING = "waiting"      # 等待开始新episode
    COLLECTING = "collecting" # 正在采集
    PAUSED = "paused"        # 暂停采集


class DataCollectionController:
    """数据采集控制器"""
    
    def __init__(self, episode_start: int = 0):
        self.current_episode = []
        self.episode_count = episode_start
        self.phase = CollectionPhase.WAITING
        self.action = None
        self.should_reset = False
        
        # 添加统计信息
        self.stats = {
            'total_frames': 0,
            'synchronized_frames': 0,
            'dropped_frames': 0
        }
        
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
    
    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == 's':
                    if self.phase in [CollectionPhase.COLLECTING, CollectionPhase.PAUSED]:
                        self.action = 'save'
                        self.phase = CollectionPhase.WAITING
                        print("\n[S] 保存当前episode并继续下一条...")
                elif key.char == 'd':
                    if self.phase != CollectionPhase.WAITING:
                        self.action = 'restart'
                        self.phase = CollectionPhase.WAITING
                        print("\n[D] 重新开始当前episode...")
                elif key.char == 'q':
                    self.action = 'quit'
                    self.phase = CollectionPhase.WAITING
                    print("\n[Q] 退出程序...")
                elif key.char == 'r':
                    if self.phase == CollectionPhase.WAITING:
                        self.should_reset = True
                        print("\n[R] 将在下个episode开始前重置系统...")
                    else:
                        print("\n[R] 采集中不能重置，将在本episode结束后执行")
                        self.should_reset = True
                elif key.char == 'i':
                    self.show_info()
            elif key == keyboard.Key.space:
                if self.phase == CollectionPhase.WAITING:
                    self.phase = CollectionPhase.COLLECTING
                    print("\n[空格] 开始采集...")
                elif self.phase == CollectionPhase.COLLECTING:
                    self.phase = CollectionPhase.PAUSED
                    print("\n[空格] 暂停采集...")
                elif self.phase == CollectionPhase.PAUSED:
                    self.phase = CollectionPhase.COLLECTING
                    print("\n[空格] 继续采集...")
        except AttributeError:
            pass
    
    def show_info(self):
        """显示详细的采集统计信息"""
        print(f"\n{'='*60}")
        print(f"当前Episode: {self.episode_count}")
        print(f"已采集帧数: {len(self.current_episode)}")
        print(f"采集阶段: {self.phase.value}")
        print(f"\n采集统计:")
        print(f"  - 总帧数: {self.stats['total_frames']}")
        print(f"  - 同步成功: {self.stats['synchronized_frames']}")
        print(f"  - 丢帧数: {self.stats['dropped_frames']}")
        if self.stats['total_frames'] > 0:
            sync_rate = (self.stats['synchronized_frames'] / self.stats['total_frames']) * 100
            print(f"  - 同步率: {sync_rate:.1f}%")
        print(f"{'='*60}\n")
    
    def stop(self):
        self.listener.stop()


class XhandMultimodalDataCollector:
    """XHand + Franka多模态数据采集接口"""
    
    def __init__(self, num_cameras: int = 1, xhand_port: str = "/dev/ttyUSB0"):
        self.num_cameras = num_cameras
        self.xhand_port = xhand_port
        self.url = "http://127.0.0.1:5000/"
        
        # 初始位姿
        self.init_pose = np.array([0.5548533772485196,0.0881488236773295,0.19591474184161564,
                                  0.7717268607706337,0.6356662995622798,0.01888957126428907,0.0030318415777171154])
        
        # 初始化独立采集器
        self.camera_collector = CameraCollector()
        self.robot_collector = RobotDataCollector(self.url)
        
        # 相机线程
        self.camera_running = False
        self.camera_thread = None
        self.robot_thread = None
        
        # 性能统计
        self.perf_stats = {
            'camera_fps': deque(maxlen=100),
            'robot_fps': deque(maxlen=100),
            'last_camera_time': 0,
            'last_robot_time': 0
        }
        
        # 初始化系统
        self._init_cameras()
        
        print("✓ XHand + Franka多模态数据采集系统初始化完成")
        print("✓ 相机分辨率: 240x240 (匹配DexUMI训练需求)")
        print("✓ 独立采集模式已启用")
        
        # 初始化重置
        print("\n执行初始化重置...")
        self.reset_to_initial_pose()
    
    def _init_cameras(self):
        """初始化相机 - 240x240分辨率"""
        top_serial = "244622072813"
        wrist_serial = "230322271519"
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) < self.num_cameras:
                logger.warning(f"只检测到 {len(devices)} 个相机，需要 {self.num_cameras} 个相机")
                self.num_cameras = min(len(devices), 1)
            
            self.pipelines = []
            cameras_initialized = 0
            
            for i in range(min(len(devices), 2)):
                if cameras_initialized >= self.num_cameras:
                    break
                    
                serial = devices[i].get_info(rs.camera_info.serial_number)
                name = devices[i].get_info(rs.camera_info.name)
                
                # 创建相机配置
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                
                # 设置分辨率：424x240便于裁剪为240x240
                config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
                # 移除深度流以减少带宽和CPU占用（未使用）
                # config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
                
                try:
                    pipeline.start(config)
                    self.pipelines.append({
                        'pipeline': pipeline,
                        'serial': serial,
                        'name': name,
                        'camera_id': cameras_initialized
                    })
                    cameras_initialized += 1
                    
                    if serial == top_serial:
                        print(f"✓ 顶部相机已初始化: {name} ({serial}) -> camera_{cameras_initialized-1}")
                    elif serial == wrist_serial:
                        print(f"✓ 腕部相机已初始化: {name} ({serial}) -> camera_{cameras_initialized-1}")
                    else:
                        print(f"✓ 相机已初始化: {name} ({serial}) -> camera_{cameras_initialized-1}")
                        
                except Exception as e:
                    logger.error(f"初始化相机 {serial} 失败: {e}")
            
            if not self.pipelines:
                raise Exception("没有相机初始化成功")
                
        except Exception as e:
            logger.error(f"初始化相机时出错: {e}")
            raise
    
    def start_collection(self):
        """启动独立的数据采集线程"""
        # 启动相机采集线程
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_thread, daemon=True)
        self.camera_thread.start()
        
        # 启动机器人数据采集线程
        self.robot_thread = threading.Thread(target=self._robot_thread, daemon=True)
        self.robot_thread.start()
        
        # 开始采集
        self.camera_collector.start_collecting()
        self.robot_collector.start_collecting()
        
        print("✓ 独立采集线程已启动")
    
    def stop_collection(self):
        """停止数据采集"""
        # 停止采集
        self.camera_collector.stop_collecting()
        self.robot_collector.stop_collecting()
        
        # 停止线程
        self.camera_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        if self.robot_thread:
            self.robot_thread.join(timeout=2.0)
        
        print("✓ 数据采集已停止")
    
    def _camera_thread(self):
        """相机采集线程 - 最大帧率采集"""
        print("相机采集线程启动")
        frame_counter = 0
        
        while self.camera_running:
            if not self.camera_collector.is_running():
                # 不采集时持续消费帧，防止缓冲区积累旧数据
                for cam_info in self.pipelines:
                    try:
                        # 使用try_wait_for_frames短时间等待并丢弃帧
                        frames = cam_info['pipeline'].try_wait_for_frames(timeout_ms=10)
                    except:
                        pass
                time.sleep(0.01)
                continue
                
            try:
                capture_start = time.time()
                frame_data = {}
                
                # 获取所有相机帧
                for cam_info in self.pipelines:
                    try:
                        # 等待新帧，设置合理超时时间
                        frames = cam_info['pipeline'].wait_for_frames(timeout_ms=50)
                        if not frames:
                            continue
                            
                        color_frame = frames.get_color_frame()
                        
                        if color_frame:
                            # 记录硬件时间戳和系统时间戳
                            hardware_timestamp = color_frame.get_timestamp()  # 硬件时间戳
                            system_timestamp = time.time()  # 系统时间戳
                            
                            # 增加帧计数器用于调试
                            frame_counter += 1
                            
                            # 获取原始图像并处理为240x240
                            # 安全的内存拷贝
                            raw_img = np.asanyarray(color_frame.get_data()).copy()  # (240, 424, 3)
                            
                            # 中心裁剪为240x240
                            h, w = raw_img.shape[:2]
                            crop_size = min(h, w)  # 240
                            start_x = (w - crop_size) // 2  # (424-240)//2 = 92
                            start_y = (h - crop_size) // 2  # 0
                            
                            # 使用copy()确保是独立的数组，不是视图
                            cropped = raw_img[start_y:start_y+crop_size, start_x:start_x+crop_size].copy()
                            
                            # 确保正确的240x240尺寸
                            if cropped.shape[:2] != (240, 240):
                                cropped = cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_AREA)
                            
                            frame_data[f'camera_{cam_info["camera_id"]}'] = {
                                'rgb': cropped,
                                'hardware_timestamp': hardware_timestamp,
                                'system_timestamp': system_timestamp,
                                'capture_time': capture_start
                            }
                    except Exception as e:
                        logger.debug(f"相机 {cam_info['camera_id']} 获取帧失败: {e}")
                
                # 验证和存储相机数据
                if frame_data:
                    # 验证帧数据有效性
                    valid_frames = {}
                    for cam_key, cam_data in frame_data.items():
                        if cam_key.startswith('camera_'):
                            cam_id = int(cam_key.split('_')[1])
                            if self.camera_collector._validate_frame(cam_data, cam_id):
                                valid_frames[cam_key] = cam_data
                            else:
                                logger.debug(f"相机 {cam_id} 帧数据验证失败")
                    
                    # 只存储验证成功的帧
                    if valid_frames:
                        with self.camera_collector.lock:
                            self.camera_collector.camera_data.append(valid_frames)
                
                # 更新FPS统计
                now = time.time()
                if self.perf_stats['last_camera_time'] > 0:
                    fps = 1.0 / (now - self.perf_stats['last_camera_time'])
                    self.perf_stats['camera_fps'].append(fps)
                self.perf_stats['last_camera_time'] = now
                
            except Exception as e:
                logger.debug(f"相机采集错误: {e}")
            
            # 控制轮询频率
            time.sleep(0.005)  # 5ms轮询间隔，在性能和稳定性间平衡
        
        print("相机采集线程停止")
    
    def _robot_thread(self):
        """机器人数据采集线程 - 固定20Hz采集"""
        print("机器人采集线程启动")
        while self.camera_running:
            if not self.robot_collector.is_running():
                time.sleep(0.01)
                continue
                
            try:
                request_start = time.time()
                
                # 获取机械臂状态
                robot_response = requests.post(self.url + "getstate", timeout=1.0)
                robot_data = robot_response.json()
                request_end = time.time()
                
                # 获取XHand触觉数据
                tactile_start = time.time()
                tactile_response = requests.post(self.url + "get_handtactile", timeout=1.0)
                tactile_data = tactile_response.json()
                tactile_end = time.time()

                # ly TODO: send object pose here from Foundationpose
                
                # 计算网络延迟
                robot_delay = request_end - request_start
                tactile_delay = tactile_end - tactile_start
                
                # 估算服务器时间戳
                robot_timestamp = request_start + robot_delay / 2
                tactile_timestamp = tactile_start + tactile_delay / 2
                
                # 构建机器人状态数据
                state_data = {
                    'tcp_pose': np.array(robot_data["pose"]),
                    'hand_action': np.array(robot_data["gripper_pos"]),
                    'proprioception': np.concatenate([np.array(robot_data["q"]), np.array(robot_data["dq"])]),
                    'velocity': np.array(robot_data["vel"]),
                    'force': np.array(robot_data["force"]),
                    'torque': np.array(robot_data["torque"]),
                    'jacobian': np.reshape(np.array(robot_data["jacobian"]), (6, 7)),
                    'fsr': np.array(tactile_data["tactile_data"]),
                    
                    # 时间戳信息
                    'request_timestamp': request_start,
                    'response_timestamp': request_end,
                    'robot_timestamp': robot_timestamp,
                    'tactile_request_timestamp': tactile_start,
                    'tactile_response_timestamp': tactile_end,
                    'tactile_timestamp': tactile_timestamp,
                    'network_delay_robot': robot_delay,
                    'network_delay_tactile': tactile_delay
                }
                
                # 存储机器人数据
                with self.robot_collector.lock:
                    self.robot_collector.robot_data.append(state_data)
                
                # 更新FPS统计
                now = time.time()
                if self.perf_stats['last_robot_time'] > 0:
                    fps = 1.0 / (now - self.perf_stats['last_robot_time'])
                    self.perf_stats['robot_fps'].append(fps)
                self.perf_stats['last_robot_time'] = now
                
                # 严格20Hz控制
                elapsed = time.time() - request_start
                target_interval = 0.05  # 50ms = 20Hz
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                
            except Exception as e:
                logger.debug(f"机器人数据采集错误: {e}")
                time.sleep(0.05)  # 错误时等待50ms
        
        print("机器人采集线程停止")
    
    def send_command(self, pos: np.ndarray):
        """发送位置命令到机械臂"""
        try:
            arr = np.array(pos[:7]).astype(np.float32)
            data = {"arr": arr.tolist()}
            response = requests.post(self.url + "pose", json=data, timeout=2.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"发送机械臂命令出错: {e}")
            return False
    
    def reset_to_initial_pose(self):
        """重置机械臂和XHand到初始位置"""
        print("正在重置系统...")
        
        success = self.send_command(self.init_pose)
        if success:
            print("  - 机械臂重置命令已发送")
        
        try:
            response = requests.post(self.url + "open_gripper", timeout=2.0)
            if response.status_code == 200:
                print("  - XHand已重置到打开状态")
        except Exception as e:
            print(f"  - XHand重置出错: {e}")
        
        time.sleep(1.5)
        print("✓ 系统重置完成")
    
    def get_display_data(self) -> Optional[Dict]:
        """获取用于实时显示的数据（不存储，只用于显示状态）"""
        # 获取最新的机器人数据
        robot_data = self.robot_collector.get_current_state()
        camera_data = self.camera_collector.get_current_frame()
        
        if not robot_data:
            return None
            
        # 构建显示数据
        display_data = {
            'tcp_pose': robot_data['tcp_pose'],
            'hand_action': robot_data['hand_action'], 
            'fsr': robot_data['fsr'],
            'camera_available': camera_data is not None
        }
        
        return display_data
    
    def get_collected_episode_data(self) -> Tuple[List[Dict], List[Dict]]:
        """获取采集完成的episode数据"""
        camera_data = self.camera_collector.get_collected_data()
        robot_data = self.robot_collector.get_collected_data()
        
        return camera_data, robot_data
    
    def clear_collected_data(self):
        """清空已采集的数据"""
        self.camera_collector.clear_data()
        self.robot_collector.clear_data()
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        camera_fps = np.mean(self.perf_stats['camera_fps']) if self.perf_stats['camera_fps'] else 0
        robot_fps = np.mean(self.perf_stats['robot_fps']) if self.perf_stats['robot_fps'] else 0
        
        return {
            'camera_fps': camera_fps,
            'robot_fps': robot_fps,
            'camera_frames': len(self.camera_collector.get_collected_data()),
            'robot_frames': len(self.robot_collector.get_collected_data())
        }
    
    def __del__(self):
        """清理资源"""
        try:
            self.stop_collection()
            for cam_info in self.pipelines:
                try:
                    cam_info['pipeline'].stop()
                except:
                    pass
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")


def save_episode_offline_aligned(episode_path: str, camera_data: List[Dict], robot_data: List[Dict]) -> bool:
    """保存离线对齐的episode数据 - 兼容现有格式"""
    if not robot_data:
        print("警告：没有机器人数据可保存")
        return False
    
    try:
        os.makedirs(episode_path, exist_ok=True)
        
        # 保存机器人数据 (保持原格式兼容性)
        tcp_poses = np.array([data["tcp_pose"] for data in robot_data])
        hand_actions = np.array([data["hand_action"] for data in robot_data])
        proprioceptions = np.array([data["proprioception"] for data in robot_data])
        fsr_data = np.array([data["fsr"] for data in robot_data])
        
        # 保存核心数据 (格式与原版完全兼容)
        with open(f"{episode_path}/pose.pkl", "wb") as f:
            pickle.dump(tcp_poses, f)
        with open(f"{episode_path}/hand_action.pkl", "wb") as f:
            pickle.dump(hand_actions, f)
        with open(f"{episode_path}/proprioception.pkl", "wb") as f:
            pickle.dump(proprioceptions, f)
        with open(f"{episode_path}/fsr.pkl", "wb") as f:
            pickle.dump(fsr_data, f)
        
        # 保存原始时间戳数据 (用于离线对齐)
        with open(f"{episode_path}/raw_timestamps.pkl", "wb") as f:
            pickle.dump({
                'robot_data': [{
                    'request_timestamp': data['request_timestamp'],
                    'response_timestamp': data['response_timestamp'], 
                    'robot_timestamp': data['robot_timestamp'],
                    'tactile_timestamp': data['tactile_timestamp'],
                    'network_delay_robot': data['network_delay_robot'],
                    'network_delay_tactile': data['network_delay_tactile']
                } for data in robot_data],
                'camera_data': camera_data  # 完整的相机时间戳信息
            }, f)
        
        # 保存相机数据 (保持原格式)
        camera_data_saved = {}
        
        # 按相机ID组织数据
        camera_frames = {}
        for frame_data in camera_data:
            for cam_key, cam_info in frame_data.items():
                if cam_key.startswith('camera_'):
                    if cam_key not in camera_frames:
                        camera_frames[cam_key] = []
                    camera_frames[cam_key].append(cam_info)
        
        # 保存每个相机的数据
        for cam_key, frames in camera_frames.items():
            cam_idx = cam_key.split('_')[1]
            cam_dir = f"{episode_path}/camera_{cam_idx}"
            os.makedirs(cam_dir, exist_ok=True)
            
            # 提取RGB数据和时间戳
            rgb_data = [frame['rgb'] for frame in frames]
            hardware_timestamps = [frame['hardware_timestamp'] for frame in frames]  
            system_timestamps = [frame['system_timestamp'] for frame in frames]
            
            if rgb_data:
                rgb_array = np.array(rgb_data)
                
                # 保存RGB数据 (兼容格式)
                with open(f"{cam_dir}/rgb.pkl", "wb") as f:
                    pickle.dump(rgb_array, f)
                
                # 保存时间戳 (兼容格式 - 使用系统时间戳)  
                with open(f"{cam_dir}/receive_time.pkl", "wb") as f:
                    pickle.dump(np.array(system_timestamps), f)
                
                # 保存原始硬件时间戳 (用于离线对齐)
                with open(f"{cam_dir}/hardware_timestamps.pkl", "wb") as f:
                    pickle.dump(np.array(hardware_timestamps), f)
                
                camera_data_saved[cam_key] = {
                    'frames': len(rgb_data),
                    'shape': rgb_array[0].shape if len(rgb_array) > 0 else None,
                    'size_mb': rgb_array.nbytes / (1024 * 1024)
                }
        
        # 创建兼容的timestamps.pkl文件 (convert_pickle_to_zarr.py需要)
        main_timestamps = np.array([data["robot_timestamp"] for data in robot_data])
        with open(f"{episode_path}/timestamps.pkl", "wb") as f:
            pickle.dump({
                'main_timestamps': main_timestamps,
                'robot_state_timestamps': main_timestamps  # 简化版本
            }, f)
        
        # 打印保存信息
        print(f"✓ 数据已保存到: {episode_path}")
        print(f"  - 机器人数据: {len(robot_data)} 帧")
        print(f"  - 相机数据:")
        for cam_name, info in camera_data_saved.items():
            print(f"    * {cam_name}: {info['frames']} 帧, 形状={info['shape']}, 大小={info['size_mb']:.2f}MB")
        print(f"  - 原始时间戳已保存，可用于后续对齐")
        
        return True
        
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")
        return False


def display_status_offline_aligned(display_data: Dict, episode_num: int, perf_stats: Dict, paused: bool = False):
    """显示采集状态信息"""
    if not display_data:
        status_text = "暂停中..." if paused else "等待数据..."
        print(f"\rEpisode {episode_num} | {status_text}", end="", flush=True)
        return
        
    tcp_pos = display_data['tcp_pose'][:3].round(3)
    fsr_norm = np.linalg.norm(display_data['fsr'], axis=1).round(2)
    camera_status = "✓" if display_data['camera_available'] else "✗"
    
    status_prefix = "[暂停] " if paused else ""
    
    print(f"\r{status_prefix}Episode {episode_num} | "
          f"TCP: {tcp_pos} | 触觉: {fsr_norm} | "
          f"相机: {camera_status} | FPS: {perf_stats.get('camera_fps', 0):.1f}/{perf_stats.get('robot_fps', 0):.1f} | "
          f"帧数: {perf_stats.get('camera_frames', 0)}/{perf_stats.get('robot_frames', 0)}", 
          end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="XHand + Franka多模态数据采集")
    parser.add_argument('--num_cameras', type=int, default=1, choices=[1, 2],
                       help='使用的相机数量 (默认: 1)')
    parser.add_argument('--data_dir', type=str, default='XhandData_Multimodal',
                       help='数据保存目录 (默认: XhandData_Multimodal)')
    parser.add_argument('--episode_start', type=int, default=None,
                       help='起始episode编号 (默认: 自动检测)')
    
    args = parser.parse_args()
    
    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 自动检测起始episode编号
    if args.episode_start is None:
        existing_episodes = []
        if os.path.exists(args.data_dir):
            for item in os.listdir(args.data_dir):
                if item.startswith('episode_'):
                    try:
                        episode_num = int(item.split('_')[1])
                        existing_episodes.append(episode_num)
                    except (IndexError, ValueError):
                        continue
        
        args.episode_start = max(existing_episodes) + 1 if existing_episodes else 0
        
        if existing_episodes:
            print(f"检测到已有 {len(existing_episodes)} 个episode，从 episode_{args.episode_start} 开始")
        else:
            print(f"未检测到已有数据，从 episode_0 开始")
    
    # 初始化采集控制器
    collector = DataCollectionController(episode_start=args.episode_start)
    
    # 初始化数据采集器
    print(f"\n初始化多模态采集系统...")
    print(f"- 相机数量: {args.num_cameras}")
    print(f"- 数据目录: {args.data_dir}")
    print(f"- 图像分辨率: 240x240 (DexUMI训练格式)")
    
    data_collector = XhandMultimodalDataCollector(num_cameras=args.num_cameras)
    
    print("\n" + "="*70)
    print("XHand + Franka多模态数据采集系统")
    print("="*70)
    print("\n✓ 系统初始化完成！")
    print("\n核心特性:")
    print("  • 独立采集 - 相机30Hz，机器人20Hz，各自最优性能")
    print("  • 高精度时间戳 - 硬件+系统双重时间戳记录")
    print("  • 离线对齐 - 消除在线同步复杂性")
    print("  • 完全兼容 - 数据格式兼容convert_pickle_to_zarr.py")
    print("  • 原始数据保留 - 支持后续时间戳分析")
    print("="*70 + "\n")
    
    try:
        while collector.action != 'quit':
            episode_path = f"{args.data_dir}/episode_{collector.episode_count}"
            collector.action = None
            
            print(f"\n{'='*60}")
            print(f"准备采集 Episode {collector.episode_count}")
            print("="*60)
            print("键盘控制:")
            print("  [空格] - 开始采集/暂停/继续")
            print("  [S] - 保存当前episode并继续下一个")  
            print("  [D] - 重新开始当前episode")
            print("  [Q] - 退出程序")
            print("  [R] - 重置系统到初始位置")
            print("  [I] - 显示详细采集信息")
            print("="*60)
            print("\n按空格键开始采集...")
            
            # 检查重置
            if collector.should_reset:
                data_collector.reset_to_initial_pose()
                collector.should_reset = False
                print("按空格键开始新的采集...")
            
            # 等待开始采集
            while collector.phase == CollectionPhase.WAITING:
                time.sleep(0.1)
                if collector.action == 'quit':
                    break
            
            if collector.action == 'quit':
                break
            
            print(f"\n开始采集 Episode {collector.episode_count}...")
            
            # 启动独立采集线程
            data_collector.start_collection()
            
            # 显示状态循环
            last_display_time = time.time()
            
            while collector.phase in [CollectionPhase.COLLECTING, CollectionPhase.PAUSED]:
                if collector.phase == CollectionPhase.PAUSED:
                    # 暂停时停止采集但保持线程运行
                    if data_collector.camera_collector.is_running():
                        print("\n[暂停] 数据采集已暂停")
                        data_collector.camera_collector.pause_collecting()
                        data_collector.robot_collector.pause_collecting()
                    time.sleep(0.1)
                    continue
                elif collector.phase == CollectionPhase.COLLECTING:
                    # 恢复采集 - 使用resume而不是start避免清空数据
                    if not data_collector.camera_collector.is_running():
                        print("\n[恢复] 数据采集已恢复")
                        data_collector.camera_collector.resume_collecting(data_collector.pipelines)
                        data_collector.robot_collector.resume_collecting()
                
                # 定期更新显示 (每100ms)
                now = time.time()
                if now - last_display_time >= 0.1:
                    display_data = data_collector.get_display_data()
                    perf_stats = data_collector.get_performance_stats()
                    
                    # 显示暂停状态
                    paused = (collector.phase == CollectionPhase.PAUSED)
                    display_status_offline_aligned(display_data, collector.episode_count, perf_stats, paused=paused)
                    last_display_time = now
                
                time.sleep(0.02)  # 50Hz显示更新
            
            # 停止采集
            data_collector.stop_collection()
            print()  # 换行
            
            # 获取采集的数据
            camera_data, robot_data = data_collector.get_collected_episode_data()
            
            # 处理采集结果
            if collector.action == 'save':
                if save_episode_offline_aligned(episode_path, camera_data, robot_data):
                    print(f"✓ Episode {collector.episode_count} 保存成功")
                    
                    # 显示最终统计
                    perf_stats = data_collector.get_performance_stats()
                    print(f"  采集统计: 相机 {len(camera_data)} 帧, 机器人 {len(robot_data)} 帧")
                    print(f"  采集性能: 相机 {perf_stats['camera_fps']:.1f} FPS, 机器人 {perf_stats['robot_fps']:.1f} FPS")
                    
                    collector.episode_count += 1
            
            elif collector.action == 'restart':
                print(f"✗ Episode {collector.episode_count} 已重置，准备重新采集")
            
            elif collector.action == 'quit':
                print(f"✗ 退出程序")
                break
            
            # 清空数据准备下一次采集
            data_collector.clear_collected_data()
    
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    
    finally:
        collector.stop()
        try:
            data_collector.stop_collection()
        except:
            pass
        del data_collector
        print("\n多模态采集系统已关闭")


if __name__ == "__main__":
    main()