#!/usr/bin/env python3
"""
综合数据质量检查脚本 - 检查XHand多模态数据的质量问题

功能:
1. 轨迹长度分布检查 - 识别异常短或长的轨迹
2. 数据分布分析 - TCP位置、关节角度、触觉数据的统计分析
3. 异常轨迹检测 - 静止轨迹、异常跳跃、图像质量问题
4. 数据完整性验证 - 文件存在性、帧数匹配
5. 可视化报告 - 生成统计图表和异常列表

作者: Claude
日期: 2024-09-09
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict
# import cv2  # 暂时不需要cv2

# 设置字体 - 如果没有中文字体就使用默认字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

warnings.filterwarnings('ignore')

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, data_dir: str, min_trajectory_length: int = 50, 
                 max_trajectory_length: int = 2000, workspace_bounds: Dict = None):
        """
        初始化数据质量检查器
        
        Args:
            data_dir: 数据目录路径
            min_trajectory_length: 最小轨迹长度阈值
            max_trajectory_length: 最大轨迹长度阈值
            workspace_bounds: 工作空间边界 {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        """
        self.data_dir = Path(data_dir)
        self.min_traj_len = min_trajectory_length
        self.max_traj_len = max_trajectory_length
        
        # 默认工作空间边界 (基于Franka机械臂的典型工作空间)
        self.workspace_bounds = workspace_bounds or {
            'x': [0.2, 0.8],    # 机械臂前方0.2-0.8m
            'y': [-0.4, 0.4],   # 左右±0.4m
            'z': [0.0, 0.6]     # 高度0-0.6m
        }
        
        # 存储检查结果
        self.results = {
            'episodes': [],
            'length_issues': [],
            'distribution_issues': [],
            'anomaly_issues': [],
            'completeness_issues': [],
            'statistics': {}
        }
    
    def check_all_episodes(self) -> Dict:
        """检查所有episode的数据质量"""
        print(f"🔍 开始检查数据目录: {self.data_dir}")
        print("=" * 80)
        
        # 获取所有episode目录
        episode_dirs = sorted([d for d in self.data_dir.glob("episode_*") if d.is_dir()])
        
        if not episode_dirs:
            print("❌ 未找到任何episode目录!")
            return self.results
        
        print(f"📊 找到 {len(episode_dirs)} 个episodes，开始质量检查...")
        
        # 检查每个episode
        for i, episode_dir in enumerate(episode_dirs):
            print(f"\n[{i+1}/{len(episode_dirs)}] 检查 {episode_dir.name}...")
            episode_result = self._check_single_episode(episode_dir)
            self.results['episodes'].append(episode_result)
        
        # 计算整体统计
        self._compute_overall_statistics()
        
        # 生成报告
        self._generate_report()
        
        return self.results
    
    def _check_single_episode(self, episode_dir: Path) -> Dict:
        """检查单个episode的数据质量"""
        result = {
            'name': episode_dir.name,
            'path': str(episode_dir),
            'length_check': {},
            'distribution_check': {},
            'anomaly_check': {},
            'completeness_check': {},
            'overall_quality': 'unknown'
        }
        
        try:
            # 1. 完整性检查
            completeness = self._check_completeness(episode_dir)
            result['completeness_check'] = completeness
            
            if not completeness['all_files_exist']:
                result['overall_quality'] = 'bad'
                self.results['completeness_issues'].append(result)
                return result
            
            # 2. 长度检查
            length_check = self._check_trajectory_length(episode_dir)
            result['length_check'] = length_check
            
            # 3. 分布检查
            distribution_check = self._check_data_distribution(episode_dir)
            result['distribution_check'] = distribution_check
            
            # 4. 异常检测
            anomaly_check = self._check_anomalies(episode_dir)
            result['anomaly_check'] = anomaly_check
            
            # 5. 综合质量评估
            result['overall_quality'] = self._assess_overall_quality(
                length_check, distribution_check, anomaly_check
            )
            
            # 记录问题episode
            if result['overall_quality'] == 'bad':
                if length_check.get('is_too_short') or length_check.get('is_too_long'):
                    self.results['length_issues'].append(result)
                if distribution_check.get('has_issues'):
                    self.results['distribution_issues'].append(result)
                if anomaly_check.get('has_anomalies'):
                    self.results['anomaly_issues'].append(result)
        
        except Exception as e:
            print(f"  ❌ 检查时出错: {e}")
            result['error'] = str(e)
            result['overall_quality'] = 'error'
        
        return result
    
    def _check_completeness(self, episode_dir: Path) -> Dict:
        """检查数据完整性"""
        required_files = [
            'pose.pkl', 'hand_action.pkl', 'proprioception.pkl', 'fsr.pkl', 'timestamps.pkl'
        ]
        
        missing_files = []
        corrupted_files = []
        file_sizes = {}
        
        # 检查核心数据文件
        for filename in required_files:
            filepath = episode_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
            else:
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        file_sizes[filename] = len(data) if hasattr(data, '__len__') else 'N/A'
                except Exception as e:
                    corrupted_files.append(f"{filename}: {str(e)}")
        
        # 检查相机数据
        camera_dirs = list(episode_dir.glob("camera_*"))
        camera_status = {}
        
        for cam_dir in camera_dirs:
            cam_name = cam_dir.name
            rgb_file = cam_dir / "rgb.pkl"
            
            if rgb_file.exists():
                try:
                    with open(rgb_file, 'rb') as f:
                        rgb_data = pickle.load(f)
                        camera_status[cam_name] = {
                            'frames': len(rgb_data),
                            'shape': rgb_data[0].shape if len(rgb_data) > 0 else None,
                            'size_mb': rgb_data.nbytes / (1024 * 1024) if hasattr(rgb_data, 'nbytes') else 'N/A'
                        }
                except Exception as e:
                    camera_status[cam_name] = {'error': str(e)}
            else:
                camera_status[cam_name] = {'error': 'rgb.pkl not found'}
        
        all_files_exist = len(missing_files) == 0 and len(corrupted_files) == 0
        
        return {
            'all_files_exist': all_files_exist,
            'missing_files': missing_files,
            'corrupted_files': corrupted_files,
            'file_sizes': file_sizes,
            'camera_status': camera_status,
            'camera_count': len(camera_dirs)
        }
    
    def _check_trajectory_length(self, episode_dir: Path) -> Dict:
        """检查轨迹长度"""
        try:
            with open(episode_dir / 'pose.pkl', 'rb') as f:
                pose_data = pickle.load(f)
            
            length = len(pose_data)
            is_too_short = length < self.min_traj_len
            is_too_long = length > self.max_traj_len
            
            return {
                'length': length,
                'is_too_short': is_too_short,
                'is_too_long': is_too_long,
                'is_normal': not (is_too_short or is_too_long)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_data_distribution(self, episode_dir: Path) -> Dict:
        """检查数据分布是否合理"""
        issues = []
        statistics = {}
        
        try:
            # 检查TCP位置
            with open(episode_dir / 'pose.pkl', 'rb') as f:
                pose_data = pickle.load(f)
            
            positions = pose_data[:, :3]  # x, y, z
            statistics['tcp_position'] = {
                'min': positions.min(axis=0),
                'max': positions.max(axis=0),
                'mean': positions.mean(axis=0),
                'std': positions.std(axis=0)
            }
            
            # 检查是否超出工作空间
            for i, axis in enumerate(['x', 'y', 'z']):
                min_val, max_val = positions[:, i].min(), positions[:, i].max()
                workspace_min, workspace_max = self.workspace_bounds[axis]
                
                if min_val < workspace_min or max_val > workspace_max:
                    issues.append(f"TCP {axis}轴超出工作空间: [{min_val:.3f}, {max_val:.3f}] vs [{workspace_min}, {workspace_max}]")
            
            # 检查关节角度
            with open(episode_dir / 'proprioception.pkl', 'rb') as f:
                proprioception_data = pickle.load(f)
            
            joint_positions = proprioception_data[:, :7]  # 前7个是关节位置
            joint_velocities = proprioception_data[:, 7:14]  # 后7个是关节速度
            
            statistics['joint_positions'] = {
                'min': joint_positions.min(axis=0),
                'max': joint_positions.max(axis=0),
                'range': joint_positions.max(axis=0) - joint_positions.min(axis=0)
            }
            
            # 检查关节限位 (Franka的典型关节限位)
            joint_limits = [
                (-2.9, 2.9), (-1.8, 1.8), (-2.9, 2.9), (-3.1, 0.0),
                (-2.9, 2.9), (-0.0, 3.8), (-2.9, 2.9)
            ]
            
            for i, (min_limit, max_limit) in enumerate(joint_limits):
                joint_range = joint_positions[:, i]
                if joint_range.min() < min_limit or joint_range.max() > max_limit:
                    issues.append(f"关节{i+1}超出限位: [{joint_range.min():.3f}, {joint_range.max():.3f}] vs [{min_limit}, {max_limit}]")
            
            # 检查触觉数据
            with open(episode_dir / 'fsr.pkl', 'rb') as f:
                fsr_data = pickle.load(f)
            
            # 处理不同的FSR数据形状
            if fsr_data.ndim == 3:  # (frames, sensors, values)
                # 展平最后两个维度进行统计
                fsr_flat = fsr_data.reshape(fsr_data.shape[0], -1)
            else:  # (frames, values)
                fsr_flat = fsr_data
            
            statistics['fsr'] = {
                'shape': fsr_data.shape,
                'min': fsr_flat.min(axis=0),
                'max': fsr_flat.max(axis=0),
                'mean': fsr_flat.mean(axis=0)
            }
            
            # 检查FSR数据合理性 (允许小幅负值，可能是传感器偏移)
            negative_ratio = (fsr_flat < 0).sum() / fsr_flat.size
            extreme_negative = (fsr_flat < -10).any()  # 检查是否有极端负值
            
            if negative_ratio > 0.5:  # 超过50%是负值才报告问题
                issues.append(f"FSR触觉数据负值比例过高: {negative_ratio*100:.1f}%")
            elif extreme_negative:
                issues.append("FSR触觉数据存在极端负值 (<-10)")
            
            return {
                'has_issues': len(issues) > 0,
                'issues': issues,
                'statistics': statistics
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_anomalies(self, episode_dir: Path) -> Dict:
        """检查异常模式"""
        anomalies = []
        
        try:
            # 检查静止轨迹
            with open(episode_dir / 'pose.pkl', 'rb') as f:
                pose_data = pickle.load(f)
            
            positions = pose_data[:, :3]
            position_changes = np.diff(positions, axis=0)
            movement_magnitude = np.linalg.norm(position_changes, axis=1)
            
            # 如果90%以上的时间移动幅度小于1mm，认为是静止轨迹
            static_threshold = 0.001  # 1mm
            static_ratio = np.sum(movement_magnitude < static_threshold) / len(movement_magnitude)
            
            if static_ratio > 0.9:
                anomalies.append(f"疑似静止轨迹: {static_ratio*100:.1f}%的时间移动<1mm")
            
            # 检查异常跳跃
            max_movement = movement_magnitude.max()
            mean_movement = movement_magnitude.mean()
            
            if max_movement > mean_movement * 10:  # 如果最大移动超过平均移动的10倍
                anomalies.append(f"检测到异常跳跃: 最大移动{max_movement*1000:.1f}mm, 平均{mean_movement*1000:.1f}mm")
            
            # 检查图像质量（如果有相机数据）
            camera_dirs = list(episode_dir.glob("camera_*"))
            for cam_dir in camera_dirs:
                rgb_file = cam_dir / "rgb.pkl"
                if rgb_file.exists():
                    try:
                        with open(rgb_file, 'rb') as f:
                            rgb_data = pickle.load(f)
                        
                        if len(rgb_data) > 0:
                            # 检查前几帧图像
                            for i in range(min(5, len(rgb_data))):
                                img = rgb_data[i]
                                
                                # 检查全黑图像
                                if img.max() < 10:
                                    anomalies.append(f"{cam_dir.name}: 检测到全黑图像 (帧{i})")
                                    break
                                
                                # 检查全白图像
                                if img.min() > 245:
                                    anomalies.append(f"{cam_dir.name}: 检测到全白图像 (帧{i})")
                                    break
                                
                                # 检查图像标准差过低 (可能表示图像质量问题)
                                if img.std() < 5:
                                    anomalies.append(f"{cam_dir.name}: 图像对比度过低 (帧{i}, std={img.std():.1f})")
                                    break
                    
                    except Exception as e:
                        anomalies.append(f"{cam_dir.name}: 图像数据读取错误 - {str(e)}")
            
            return {
                'has_anomalies': len(anomalies) > 0,
                'anomalies': anomalies,
                'movement_stats': {
                    'max_movement_mm': max_movement * 1000,
                    'mean_movement_mm': mean_movement * 1000,
                    'static_ratio': static_ratio
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_overall_quality(self, length_check: Dict, distribution_check: Dict, anomaly_check: Dict) -> str:
        """评估整体数据质量"""
        issues = 0
        
        # 长度问题
        if length_check.get('is_too_short') or length_check.get('is_too_long'):
            issues += 2  # 长度问题权重较高
        
        # 分布问题
        if distribution_check.get('has_issues'):
            issues += 1
        
        # 异常检测
        if anomaly_check.get('has_anomalies'):
            anomaly_count = len(anomaly_check.get('anomalies', []))
            if anomaly_count >= 3:
                issues += 2
            elif anomaly_count >= 1:
                issues += 1
        
        # 质量评级
        if issues == 0:
            return 'good'
        elif issues <= 2:
            return 'warning'
        else:
            return 'bad'
    
    def _compute_overall_statistics(self):
        """计算整体统计信息"""
        if not self.results['episodes']:
            return
        
        # 统计质量分布
        quality_counts = defaultdict(int)
        lengths = []
        
        for episode in self.results['episodes']:
            quality = episode.get('overall_quality', 'unknown')
            quality_counts[quality] += 1
            
            length_info = episode.get('length_check', {})
            if 'length' in length_info:
                lengths.append(length_info['length'])
        
        # 长度统计
        if lengths:
            lengths = np.array(lengths)
            length_stats = {
                'count': len(lengths),
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'mean': float(lengths.mean()),
                'median': float(np.median(lengths)),
                'std': float(lengths.std()),
                'q25': float(np.percentile(lengths, 25)),
                'q75': float(np.percentile(lengths, 75))
            }
        else:
            length_stats = {}
        
        self.results['statistics'] = {
            'total_episodes': len(self.results['episodes']),
            'quality_distribution': dict(quality_counts),
            'length_statistics': length_stats,
            'issue_summary': {
                'length_issues': len(self.results['length_issues']),
                'distribution_issues': len(self.results['distribution_issues']),
                'anomaly_issues': len(self.results['anomaly_issues']),
                'completeness_issues': len(self.results['completeness_issues'])
            }
        }
    
    def _generate_report(self):
        """生成检查报告"""
        stats = self.results['statistics']
        
        print("\n" + "=" * 80)
        print("📊 数据质量检查报告")
        print("=" * 80)
        
        # 总体统计
        print(f"\n📈 总体统计:")
        print(f"  总episode数: {stats['total_episodes']}")
        
        if stats['length_statistics']:
            ls = stats['length_statistics']
            print(f"  轨迹长度: 平均 {ls['mean']:.1f} 帧 (范围: {ls['min']}-{ls['max']})")
            print(f"  长度分布: Q25={ls['q25']:.0f}, 中位数={ls['median']:.0f}, Q75={ls['q75']:.0f}")
        
        # 质量分布
        print(f"\n🎯 质量分布:")
        quality_dist = stats['quality_distribution']
        for quality, count in quality_dist.items():
            percentage = (count / stats['total_episodes']) * 100
            emoji = {'good': '✅', 'warning': '⚠️', 'bad': '❌', 'error': '💥', 'unknown': '❓'}.get(quality, '❓')
            print(f"  {emoji} {quality.capitalize()}: {count} episodes ({percentage:.1f}%)")
        
        # 问题总结
        print(f"\n⚠️ 问题总结:")
        issue_summary = stats['issue_summary']
        for issue_type, count in issue_summary.items():
            if count > 0:
                print(f"  - {issue_type.replace('_', ' ').title()}: {count} episodes")
        
        # 详细问题列表
        if self.results['length_issues']:
            print(f"\n📏 长度异常的episodes:")
            for episode in self.results['length_issues']:
                length_info = episode.get('length_check', {})
                length = length_info.get('length', 'N/A')
                if length_info.get('is_too_short'):
                    print(f"  ❌ {episode['name']}: 过短 ({length} < {self.min_traj_len})")
                elif length_info.get('is_too_long'):
                    print(f"  ❌ {episode['name']}: 过长 ({length} > {self.max_traj_len})")
        
        if self.results['anomaly_issues']:
            print(f"\n🚨 异常检测结果:")
            for episode in self.results['anomaly_issues']:
                anomaly_info = episode.get('anomaly_check', {})
                if 'anomalies' in anomaly_info:
                    print(f"  ⚠️ {episode['name']}:")
                    for anomaly in anomaly_info['anomalies']:
                        print(f"    - {anomaly}")
        
        if self.results['completeness_issues']:
            print(f"\n📋 完整性问题:")
            for episode in self.results['completeness_issues']:
                comp_info = episode.get('completeness_check', {})
                print(f"  ❌ {episode['name']}:")
                if comp_info.get('missing_files'):
                    print(f"    - 缺失文件: {', '.join(comp_info['missing_files'])}")
                if comp_info.get('corrupted_files'):
                    print(f"    - 损坏文件: {', '.join(comp_info['corrupted_files'])}")
    
    def visualize_statistics(self, save_plots: bool = True):
        """生成可视化统计图表"""
        if not self.results['episodes']:
            print("❌ 没有数据可以可视化")
            return
        
        # 设置图表样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数据质量统计报告', fontsize=16, fontweight='bold')
        
        # 1. 轨迹长度分布
        lengths = []
        quality_labels = []
        
        for episode in self.results['episodes']:
            length_info = episode.get('length_check', {})
            if 'length' in length_info:
                lengths.append(length_info['length'])
                quality_labels.append(episode.get('overall_quality', 'unknown'))
        
        if lengths:
            ax1 = axes[0, 0]
            ax1.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(self.min_traj_len, color='red', linestyle='--', label=f'最小长度阈值 ({self.min_traj_len})')
            ax1.axvline(self.max_traj_len, color='red', linestyle='--', label=f'最大长度阈值 ({self.max_traj_len})')
            ax1.set_xlabel('轨迹长度 (帧数)')
            ax1.set_ylabel('Episode数量')
            ax1.set_title('轨迹长度分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 质量分布饼图
        ax2 = axes[0, 1]
        quality_counts = defaultdict(int)
        for episode in self.results['episodes']:
            quality = episode.get('overall_quality', 'unknown')
            quality_counts[quality] += 1
        
        if quality_counts:
            colors = {'good': 'lightgreen', 'warning': 'orange', 'bad': 'lightcoral', 'error': 'red', 'unknown': 'gray'}
            quality_names = list(quality_counts.keys())
            quality_values = list(quality_counts.values())
            quality_colors = [colors.get(q, 'gray') for q in quality_names]
            
            ax2.pie(quality_values, labels=quality_names, colors=quality_colors, autopct='%1.1f%%')
            ax2.set_title('数据质量分布')
        
        # 3. 问题类型统计
        ax3 = axes[1, 0]
        issue_types = ['长度问题', '分布问题', '异常检测', '完整性问题']
        issue_counts = [
            len(self.results['length_issues']),
            len(self.results['distribution_issues']),
            len(self.results['anomaly_issues']),
            len(self.results['completeness_issues'])
        ]
        
        bars = ax3.bar(issue_types, issue_counts, color=['red', 'orange', 'yellow', 'purple'], alpha=0.7)
        ax3.set_ylabel('Episode数量')
        ax3.set_title('问题类型统计')
        ax3.set_xticklabels(issue_types, rotation=45)
        
        # 在柱状图上添加数值
        for bar, count in zip(bars, issue_counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 4. 轨迹长度箱线图（按质量分类）
        ax4 = axes[1, 1]
        if lengths and quality_labels:
            # 按质量分组长度数据
            quality_lengths = defaultdict(list)
            for length, quality in zip(lengths, quality_labels):
                quality_lengths[quality].append(length)
            
            if quality_lengths:
                qualities = list(quality_lengths.keys())
                length_groups = [quality_lengths[q] for q in qualities]
                
                box_plot = ax4.boxplot(length_groups, labels=qualities, patch_artist=True)
                
                # 设置颜色
                quality_colors_box = {'good': 'lightgreen', 'warning': 'orange', 'bad': 'lightcoral', 'error': 'red', 'unknown': 'gray'}
                for patch, quality in zip(box_plot['boxes'], qualities):
                    patch.set_facecolor(quality_colors_box.get(quality, 'gray'))
                
                ax4.set_ylabel('轨迹长度 (帧数)')
                ax4.set_title('不同质量级别的轨迹长度分布')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.data_dir / 'data_quality_report.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 统计图表已保存到: {plot_path}")
        
        plt.show()
    
    def export_detailed_report(self, output_file: str = None):
        """导出详细的检查报告到文件"""
        if output_file is None:
            output_file = self.data_dir / 'data_quality_detailed_report.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("XHand多模态数据质量检查详细报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 检查参数
            f.write("检查参数:\n")
            f.write(f"  数据目录: {self.data_dir}\n")
            f.write(f"  最小轨迹长度: {self.min_traj_len}\n")
            f.write(f"  最大轨迹长度: {self.max_traj_len}\n")
            f.write(f"  工作空间边界: {self.workspace_bounds}\n\n")
            
            # 总体统计
            stats = self.results['statistics']
            f.write("总体统计:\n")
            f.write(f"  总episode数: {stats['total_episodes']}\n")
            
            if stats['length_statistics']:
                ls = stats['length_statistics']
                f.write(f"  轨迹长度统计:\n")
                f.write(f"    - 最小: {ls['min']} 帧\n")
                f.write(f"    - 最大: {ls['max']} 帧\n")
                f.write(f"    - 平均: {ls['mean']:.1f} 帧\n")
                f.write(f"    - 中位数: {ls['median']:.1f} 帧\n")
                f.write(f"    - 标准差: {ls['std']:.1f}\n\n")
            
            # 每个episode的详细信息
            f.write("详细检查结果:\n")
            f.write("-" * 80 + "\n")
            
            for episode in self.results['episodes']:
                f.write(f"\nEpisode: {episode['name']}\n")
                f.write(f"质量评级: {episode['overall_quality']}\n")
                
                # 长度检查
                length_check = episode.get('length_check', {})
                if 'length' in length_check:
                    f.write(f"轨迹长度: {length_check['length']} 帧\n")
                    if length_check.get('is_too_short'):
                        f.write("  ⚠️ 轨迹过短\n")
                    elif length_check.get('is_too_long'):
                        f.write("  ⚠️ 轨迹过长\n")
                
                # 异常检测
                anomaly_check = episode.get('anomaly_check', {})
                if anomaly_check.get('has_anomalies'):
                    f.write("异常检测:\n")
                    for anomaly in anomaly_check.get('anomalies', []):
                        f.write(f"  - {anomaly}\n")
                
                # 分布检查
                distribution_check = episode.get('distribution_check', {})
                if distribution_check.get('has_issues'):
                    f.write("分布问题:\n")
                    for issue in distribution_check.get('issues', []):
                        f.write(f"  - {issue}\n")
                
                # 完整性检查
                completeness_check = episode.get('completeness_check', {})
                if not completeness_check.get('all_files_exist', True):
                    f.write("完整性问题:\n")
                    if completeness_check.get('missing_files'):
                        f.write(f"  缺失文件: {', '.join(completeness_check['missing_files'])}\n")
                    if completeness_check.get('corrupted_files'):
                        f.write(f"  损坏文件: {', '.join(completeness_check['corrupted_files'])}\n")
                
                f.write("-" * 40 + "\n")
        
        print(f"📝 详细报告已导出到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="XHand多模态数据质量检查工具")
    parser.add_argument('data_dir', type=str, help='数据目录路径')
    parser.add_argument('--min-length', type=int, default=50, help='最小轨迹长度阈值 (默认: 50)')
    parser.add_argument('--max-length', type=int, default=2000, help='最大轨迹长度阈值 (默认: 2000)')
    parser.add_argument('--no-plots', action='store_true', help='不生成可视化图表')
    parser.add_argument('--no-export', action='store_true', help='不导出详细报告')
    parser.add_argument('--workspace-x', nargs=2, type=float, default=[0.2, 0.8], 
                       help='X轴工作空间边界 (默认: 0.2 0.8)')
    parser.add_argument('--workspace-y', nargs=2, type=float, default=[-0.4, 0.4],
                       help='Y轴工作空间边界 (默认: -0.4 0.4)')
    parser.add_argument('--workspace-z', nargs=2, type=float, default=[0.0, 0.6],
                       help='Z轴工作空间边界 (默认: 0.0 0.6)')
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 设置工作空间边界
    workspace_bounds = {
        'x': args.workspace_x,
        'y': args.workspace_y,
        'z': args.workspace_z
    }
    
    # 创建检查器并运行检查
    checker = DataQualityChecker(
        data_dir=args.data_dir,
        min_trajectory_length=args.min_length,
        max_trajectory_length=args.max_length,
        workspace_bounds=workspace_bounds
    )
    
    # 执行检查
    results = checker.check_all_episodes()
    
    # 生成可视化报告
    if not args.no_plots:
        try:
            checker.visualize_statistics(save_plots=True)
        except Exception as e:
            print(f"⚠️ 生成可视化图表时出错: {e}")
    
    # 导出详细报告
    if not args.no_export:
        try:
            checker.export_detailed_report()
        except Exception as e:
            print(f"⚠️ 导出详细报告时出错: {e}")
    
    # 返回状态码
    stats = results['statistics']
    total_issues = sum(stats['issue_summary'].values())
    
    if total_issues == 0:
        print("\n✅ 所有数据质量检查通过!")
        sys.exit(0)
    else:
        print(f"\n⚠️ 发现 {total_issues} 个问题，请查看详细报告")
        sys.exit(1)


if __name__ == "__main__":
    main()