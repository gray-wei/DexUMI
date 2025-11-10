# XHand多模态数据质量检查工具使用说明

## 概述

`comprehensive_data_quality_check.py` 是一个全面的数据质量检查工具，用于验证XHand多模态数据采集的质量。它可以检测多种数据问题，包括轨迹长度异常、数据分布问题、异常模式和数据完整性问题。

## 主要功能

### 1. 轨迹长度检查
- 检测过短的轨迹（可能是意外保存的数据）
- 检测过长的轨迹（可能是忘记停止采集）
- 统计轨迹长度分布

### 2. 数据分布分析
- TCP位置是否在合理的工作空间内
- 关节角度是否超出机械臂限位
- 触觉数据是否合理（非负值）

### 3. 异常检测
- 静止轨迹检测（机械臂基本不动）
- 异常跳跃检测（位置突然大幅变化）
- 图像质量问题（全黑、全白、对比度过低）

### 4. 数据完整性验证
- 必要文件是否存在
- pickle文件是否损坏
- 不同数据源的帧数是否匹配

### 5. 可视化报告
- 轨迹长度分布直方图
- 数据质量分布饼图
- 问题类型统计
- 分质量级别的箱线图

## 安装依赖

```bash
pip install numpy matplotlib seaborn opencv-python
```

## 基本用法

### 快速检查
```bash
# 检查数据目录（使用默认参数）
python comprehensive_data_quality_check.py /path/to/your/XhandData_Multimodal

# 示例：检查当前目录下的数据
python comprehensive_data_quality_check.py ./XhandData_Multimodal
```

### 自定义参数
```bash
# 设置自定义的轨迹长度阈值
python comprehensive_data_quality_check.py ./XhandData_Multimodal \
    --min-length 30 \
    --max-length 1500

# 设置自定义工作空间边界
python comprehensive_data_quality_check.py ./XhandData_Multimodal \
    --workspace-x 0.3 0.7 \
    --workspace-y -0.3 0.3 \
    --workspace-z 0.1 0.5
```

### 禁用某些功能
```bash
# 不生成可视化图表
python comprehensive_data_quality_check.py ./XhandData_Multimodal --no-plots

# 不导出详细报告文件
python comprehensive_data_quality_check.py ./XhandData_Multimodal --no-export

# 只进行基本检查
python comprehensive_data_quality_check.py ./XhandData_Multimodal --no-plots --no-export
```

## 输出文件

运行检查后，工具会在数据目录下生成以下文件：

1. **data_quality_report.png** - 可视化统计图表
2. **data_quality_detailed_report.txt** - 详细的文本报告

## 报告解读

### 质量等级
- **✅ Good**: 数据质量良好，无明显问题
- **⚠️ Warning**: 有轻微问题，但可以使用
- **❌ Bad**: 有严重问题，建议重新采集
- **💥 Error**: 数据读取出错
- **❓ Unknown**: 无法确定质量

### 常见问题类型

#### 长度问题
- **过短轨迹**: 通常<50帧，可能是意外保存
- **过长轨迹**: 通常>2000帧，可能忘记停止采集

#### 分布问题
- **超出工作空间**: TCP位置超出机械臂安全工作范围
- **关节限位违反**: 关节角度超出硬件限制
- **触觉数据异常**: FSR传感器数据异常（如负值）

#### 异常检测
- **静止轨迹**: 90%以上时间移动<1mm
- **异常跳跃**: 位置突变超过平均移动的10倍
- **图像质量**: 全黑、全白或对比度过低的图像

#### 完整性问题
- **缺失文件**: 缺少必要的数据文件
- **损坏文件**: pickle文件无法正常读取
- **帧数不匹配**: 不同数据源的帧数不一致

## 使用示例

### 示例1：检查新采集的数据
```bash
# 检查刚采集的数据，使用较宽松的参数
python comprehensive_data_quality_check.py ./XhandData_Multimodal \
    --min-length 20 \
    --max-length 3000
```

### 示例2：严格质量控制
```bash
# 用于训练前的严格质量检查
python comprehensive_data_quality_check.py ./XhandData_Multimodal \
    --min-length 100 \
    --max-length 1000 \
    --workspace-x 0.4 0.7 \
    --workspace-y -0.2 0.2
```

### 示例3：快速批量检查
```bash
# 快速检查多个数据集（脚本化使用）
for dataset in dataset1 dataset2 dataset3; do
    echo "检查 $dataset..."
    python comprehensive_data_quality_check.py ./data/$dataset --no-plots --no-export
done
```

## 质量控制建议

### 数据采集时
1. 确保轨迹长度适中（50-1000帧为佳）
2. 保持机械臂在安全工作空间内操作
3. 避免过度静止或剧烈运动
4. 检查相机视野和光照条件

### 数据后处理
1. 删除质量为"Bad"的episodes
2. 对"Warning"级别的数据进行人工审查
3. 确保训练集中的数据分布均匀
4. 保留原始数据的备份

### 训练前验证
1. 运行完整的质量检查
2. 确保至少80%的数据质量为"Good"
3. 检查数据集的整体分布
4. 验证与训练需求的匹配性

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | - | 数据目录路径（必需） |
| `--min-length` | 50 | 最小轨迹长度阈值 |
| `--max-length` | 2000 | 最大轨迹长度阈值 |
| `--workspace-x` | [0.2, 0.8] | X轴工作空间边界 |
| `--workspace-y` | [-0.4, 0.4] | Y轴工作空间边界 |
| `--workspace-z` | [0.0, 0.6] | Z轴工作空间边界 |
| `--no-plots` | False | 禁用可视化图表生成 |
| `--no-export` | False | 禁用详细报告导出 |

## 返回状态码

- **0**: 所有数据质量检查通过
- **1**: 发现质量问题，需要人工审查

## 故障排除

### 常见错误

1. **ImportError**: 缺少依赖库
   ```bash
   pip install numpy matplotlib seaborn opencv-python
   ```

2. **FileNotFoundError**: 数据目录不存在
   - 检查路径是否正确
   - 确保目录包含episode_*子目录

3. **PermissionError**: 权限不足
   - 检查文件和目录的读写权限
   - 使用sudo或更改文件所有权

4. **MemoryError**: 内存不足
   - 处理大数据集时可能发生
   - 考虑分批处理或增加系统内存

### 性能优化

1. **大数据集处理**：
   - 使用`--no-plots`跳过可视化
   - 分批处理多个子目录

2. **网络存储**：
   - 将数据复制到本地SSD
   - 使用快速网络连接

## 与其他工具的集成

### 与现有检查脚本的关系
- `check_data_consistency.py`: 专注于帧数一致性检查
- `validate_dexumi_data.py`: 专注于zarr格式验证
- `comprehensive_data_quality_check.py`: 提供全面的质量分析

### 工作流程建议
1. 数据采集后立即运行基本检查
2. 使用comprehensive工具进行深度分析
3. 转换为zarr格式前进行最终验证
4. 训练前再次确认数据质量

## 技术支持

如有问题或建议，请：
1. 检查错误消息和日志
2. 确认数据格式与XhandMultimodalCollection.py兼容
3. 查看生成的详细报告文件
4. 参考代码注释了解检查逻辑

---

*最后更新: 2024-09-09*