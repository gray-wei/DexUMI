# eval_xhand_franka.py 修改说明

## 主要修改内容

### 1. 移除不需要的组件
- ✅ 移除 Spacemouse（3D鼠标）相关代码
- ✅ 移除 UR5eClient，替换为 HTTPRobotClient
- ✅ 移除手动控制循环（原 line 238-310）
- ✅ 移除未使用的变量（match_initial_frame 等）

### 2. 坐标系简化
- ✅ 移除 T_ET 变换矩阵定义
- ✅ 简化坐标变换逻辑：
  - 原代码：`T_BN = T_BE @ T_ET @ relative_pose @ T_ET^(-1)`
  - 新代码：`T_BN = T_BE @ relative_pose`
- 原理：因为直接使用 Franka ee_pose，不需要工具坐标系转换

### 3. HTTP 控制接口
- ✅ 使用 HTTPRobotClient 控制机器人
- ✅ 使用 HTTPHandClient 控制灵巧手
- ✅ 处理 7D (xyz + quaternion) 到 6D (xyz + rotvec) 的转换
- ✅ 兼容 HTTP 服务器返回的姿态格式

### 4. 固定初始位置
- ✅ 设置固定的机器人初始位置（与数据采集一致）
- ✅ 启动时自动移动到初始位置，无需手动调整
- ✅ 初始位置取自 XhandMultimodalCollection.py

### 5. 相机系统支持
- ✅ 支持 RealSense 相机（新增）
- ✅ 保留 OAK 相机兼容性（可选）
- ✅ 添加相机类型选择参数 `--camera_type`
- ✅ 自动处理图像大小为 240x240（与训练数据一致）

### 6. 保留的核心功能
- ✅ Diffusion Policy 推理逻辑
- ✅ 多步预测和部分执行（exec_horizon）
- ✅ 时序控制和延迟补偿
- ✅ FSR 传感器数据处理
- ✅ 姿态插值（获取图像拍摄时刻的机器人姿态）
- ✅ 视频录制功能
- ✅ 相对/绝对手部动作转换

## 使用方法

```bash
# 使用 RealSense 相机
python real_script/eval_policy/eval_xhand_franka.py \
    --model_path /path/to/model \
    --ckpt 600 \
    --frequency 10 \
    --exec_horizon 8 \
    --camera_type realsense \
    --camera_latency 0.185 \
    --hand_action_latency 0.3 \
    --robot_action_latency 0.170

# 使用 OAK 相机（兼容模式）
python real_script/eval_policy/eval_xhand_franka.py \
    --model_path /path/to/model \
    --ckpt 600 \
    --camera_type oak \
    --frequency 10 \
    --exec_horizon 8
```

## 命令行参数

- `--frequency`: 控制频率 (Hz)，默认 10
- `--camera_type`: 相机类型选择 ['realsense', 'oak']，默认 realsense
- `--enable_record_camera`: 启用录制相机（需要第二个相机）
- `--model_path`: 模型文件路径
- `--ckpt`: 检查点编号
- `--camera_latency`: 相机延迟（秒），默认 0.185
- `--hand_action_latency`: 手部动作延迟（秒），默认 0.3
- `--robot_action_latency`: 机器人动作延迟（秒），默认 0.170
- `--exec_horizon`: 执行时间窗口，默认 8
- `--video_record_path`: 视频录制路径
- `--match_episode_path`: 匹配 episode 路径（可选）

## 注意事项

1. **HTTP 服务器**：确保 franka_server.py 在 http://127.0.0.1:5000 运行
2. **相机配置**：
   - RealSense：自动检测并使用第一个可用设备
   - OAK：使用原有的 OAK 相机接口
   - 输出图像自动调整为 240x240
3. **坐标系一致性**：确保训练数据也是在末端执行器坐标系下
4. **初始位置**：initial_robot_pose 需要根据实际数据采集时的位置调整

## 相对控制语义

保持了原有的相对控制语义：
- 策略输出：相对变换（delta）
- 执行方式：当前姿态 + 相对增量 = 目标姿态
- 发送命令：绝对目标位置

简化后的公式更直观：
```python
# 原始复杂版本（适用于 iPhone wrist pose）
T_BN = T_BE @ T_ET @ relative_pose @ T_ET^(-1)

# 简化版本（直接使用 Franka ee_pose）
T_BN = T_BE @ relative_pose
```

其中：
- `T_BE`：当前末端执行器姿态（基座坐标系）
- `relative_pose`：策略输出的相对变换
- `T_BN`：目标姿态（基座坐标系）
- `T_ET`：末端执行器到工具的变换（已移除）

## 实现完成状态

✅ **已完成**：
1. 所有代码修改和简化
2. RealSense 相机集成
3. HTTP 控制接口替换
4. 坐标系简化
5. 固定初始位置设置
6. 未使用变量清理

⚠️ **待测试**：
1. 实际硬件连接测试
2. 策略模型推理验证
3. 时序控制精度
4. RealSense 相机实时性能

📝 **可能需要调整**：
1. 初始机器人位置（根据实际数据采集设置）
2. 各种延迟参数（根据实际硬件测量）
3. FSR 传感器阈值（根据实际硬件校准）