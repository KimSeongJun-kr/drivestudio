#!/usr/bin/env python3
"""
인스턴스별 오차 데이터를 시각화하는 그래프 생성 스크립트

사용법:
    python write_instancewise_eval_graph.py --input_json path/to/instance_wise_frame_err_data.json --output_dir path/to/output/
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any, Optional
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 폰트 크기 설정 (전체적으로 크기 증가)
plt.rcParams['font.size'] = 16          # 기본 폰트 크기
plt.rcParams['axes.titlesize'] = 20     # 제목 크기
plt.rcParams['axes.labelsize'] = 18     # x, y축 라벨 크기
plt.rcParams['xtick.labelsize'] = 16    # x축 틱 라벨 크기
plt.rcParams['ytick.labelsize'] = 16    # y축 틱 라벨 크기
plt.rcParams['legend.fontsize'] = 16    # 범례 폰트 크기
plt.rcParams['figure.titlesize'] = 22   # 전체 그림 제목 크기

OBJECT_CLASS_NODE_MAPPING = {
    # Rigid objects (vehicles)
    "vehicle.bus.bendy": "RigidNodes",
    "vehicle.bus.rigid": "RigidNodes",
    "vehicle.car": "RigidNodes",
    "vehicle.construction": "RigidNodes",
    "vehicle.emergency.ambulance": "RigidNodes",
    "vehicle.emergency.police": "RigidNodes",
    "vehicle.motorcycle": "RigidNodes",
    "vehicle.trailer": "RigidNodes",
    "vehicle.truck": "RigidNodes",

    # Humans (SMPL model)
    "human.pedestrian.adult": "SMPLNodes",
    "human.pedestrian.child": "SMPLNodes",
    "human.pedestrian.construction_worker": "SMPLNodes",
    "human.pedestrian.police_officer": "SMPLNodes",

    # Potentially deformable objects
    "human.pedestrian.personal_mobility": "DeformableNodes",
    "human.pedestrian.stroller": "DeformableNodes",
    "human.pedestrian.wheelchair": "DeformableNodes",
    "animal": "DeformableNodes",
    "vehicle.bicycle": "DeformableNodes"
}

detection_mapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
detection_mapping_inv = {v: k for k, v in detection_mapping.items()}

def load_json_data(json_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_data_for_plotting(data: Dict[str, Any], target_node_class: List[str]) -> Dict[str, List]:
    """그래프 그리기 위한 데이터를 추출합니다."""
    plot_data = {
        'instance_ids': [],
        'instance_tokens': [],
        'frames': [],
        'trans_err': [],
        'vel_err': [],
        'scale_err': [],
        'orient_err': [],
        'attr_err': [],
        'dist': [],
        'num_gaussians': [],
        'detection_name': []
    }
    
    for instance_id, instance_data in data.items():
        num_frames = len(instance_data['frames'])
        if num_frames != len(instance_data['trans_err']) or num_frames != len(instance_data['vel_err']) or num_frames != len(instance_data['scale_err']) or num_frames != len(instance_data['orient_err']) or num_frames != len(instance_data['attr_err']) or num_frames != len(instance_data['dist']) :
            print(f"Error: data length mismatch: {instance_id}")
            continue

        node_class = OBJECT_CLASS_NODE_MAPPING[detection_mapping_inv[instance_data['detection_name']]]
        if node_class not in target_node_class:
            continue
        
        # instance_token 추출 (있는 경우)
        instance_token = instance_data.get('instance_token', instance_id)
        
        # 각 프레임별 데이터 추가
        plot_data['instance_ids'].extend([instance_data['instance_idx']] * num_frames)
        plot_data['instance_tokens'].extend([instance_token] * num_frames)
        plot_data['frames'].extend(instance_data['frames'])
        plot_data['trans_err'].extend(instance_data['trans_err'])
        plot_data['vel_err'].extend(instance_data['vel_err'])
        plot_data['scale_err'].extend(instance_data['scale_err'])
        plot_data['orient_err'].extend(instance_data['orient_err'])
        plot_data['attr_err'].extend(instance_data['attr_err'])
        plot_data['dist'].extend(instance_data['dist'])
        plot_data['num_gaussians'].extend([instance_data['num_gaussians']] * num_frames)
        plot_data['detection_name'].extend([instance_data['detection_name']] * num_frames)
    return plot_data

def calculate_mae_per_instance(plot_data: Dict[str, List]) -> List[tuple]:
    """각 인스턴스별로 trans_err의 MAE를 계산하고 MAE 순으로 정렬된 리스트를 반환합니다."""
    unique_instances = list(set(plot_data['instance_ids']))
    instance_mae_list = []
    
    for instance_id in unique_instances:
        instance_mask = [inst_id == instance_id for inst_id in plot_data['instance_ids']]
        instance_trans_errors = [e for e, mask in zip(plot_data['trans_err'], instance_mask) if mask]
        
        if instance_trans_errors:
            mae = np.mean(np.abs(instance_trans_errors))
            instance_mae_list.append((instance_id, mae))
    
    # MAE 기준으로 오름차순 정렬
    instance_mae_list.sort(key=lambda x: x[1])
    return instance_mae_list

def group_instances_by_mae(instance_mae_list: List[tuple], group_size: int = 10) -> List[List[str]]:
    """MAE 순으로 정렬된 인스턴스를 지정된 크기로 그룹화합니다."""
    if group_size == 0:
        # group_size가 0이면 모든 인스턴스를 하나의 그룹으로 만듦
        all_instances = [instance_id for instance_id, _ in instance_mae_list]
        return [all_instances]
    
    groups = []
    for i in range(0, len(instance_mae_list), group_size):
        group = [instance_id for instance_id, _ in instance_mae_list[i:i + group_size]]
        groups.append(group)
    return groups

def calculate_max_values_for_axes(plot_data: Dict[str, List], initial_plot_data: Optional[Dict[str, List]] = None) -> Dict[str, float]:
    """전체 데이터에서 각 축별 최대값을 계산합니다."""
    error_types = ['trans_err', 'scale_err', 'orient_err']
    x_axis_types = ['frames', 'dist', 'num_gaussians']
    max_values = {}
    
    # y축 (오차 타입) 최대값 계산
    for error_type in error_types:
        # 현재 데이터의 최대값
        current_max = max(plot_data[error_type]) if plot_data[error_type] else 0
        
        # 초기값 데이터가 있으면 함께 고려
        if initial_plot_data and initial_plot_data[error_type]:
            initial_max = max(initial_plot_data[error_type])
            max_values[error_type] = max(current_max, initial_max)
        else:
            max_values[error_type] = current_max
    
    # x축 데이터 최대값 계산
    for x_axis_type in x_axis_types:
        # 현재 데이터의 최대값
        current_max = max(plot_data[x_axis_type]) if plot_data[x_axis_type] else 0
        
        # 초기값 데이터가 있으면 함께 고려
        if initial_plot_data and initial_plot_data[x_axis_type]:
            initial_max = max(initial_plot_data[x_axis_type])
            max_values[x_axis_type] = max(current_max, initial_max)
        else:
            max_values[x_axis_type] = current_max
    
    return max_values

def load_and_process_initial_data(initial_json_path: str, target_node_class: List[str]) -> Optional[Dict[str, List]]:
    """초기값 오차 데이터를 로드하고 처리합니다."""
    if not initial_json_path or not os.path.exists(initial_json_path):
        return None
    
    print(f"초기값 JSON 파일 로딩: {initial_json_path}")
    initial_data = load_json_data(initial_json_path)
    initial_plot_data = extract_data_for_plotting(initial_data, target_node_class)
    return initial_plot_data

def create_token_to_data_mapping(plot_data: Dict[str, List]) -> Dict[str, Dict]:
    """instance_token을 키로 하는 데이터 매핑을 생성합니다."""
    token_mapping = {}
    
    for i in range(len(plot_data['instance_tokens'])):
        token = plot_data['instance_tokens'][i]
        
        if token not in token_mapping:
            token_mapping[token] = {
                'instance_id': plot_data['instance_ids'][i],
                'frames': [],
                'trans_err': [],
                'vel_err': [],
                'scale_err': [],
                'orient_err': [],
                'attr_err': [],
                'dist': [],
                'num_gaussians': [],
                'detection_name': []
            }
        
        # 각 프레임별 데이터 추가
        token_mapping[token]['frames'].append(plot_data['frames'][i])
        token_mapping[token]['trans_err'].append(plot_data['trans_err'][i])
        token_mapping[token]['vel_err'].append(plot_data['vel_err'][i])
        token_mapping[token]['scale_err'].append(plot_data['scale_err'][i])
        token_mapping[token]['orient_err'].append(plot_data['orient_err'][i])
        token_mapping[token]['attr_err'].append(plot_data['attr_err'][i])
        token_mapping[token]['dist'].append(plot_data['dist'][i])
        token_mapping[token]['num_gaussians'].append(plot_data['num_gaussians'][i])
        token_mapping[token]['detection_name'].append(plot_data['detection_name'][i])

    return token_mapping

def filter_initial_data_by_frames(initial_data: Dict[str, Any], target_frames: List[int]) -> Dict[str, Any]:
    """초기값 데이터에서 target_frames에 해당하는 프레임만 필터링합니다."""
    filtered_data = {
        'frames': [],
        'trans_err': [],
        'vel_err': [],
        'scale_err': [],
        'orient_err': [],
        'attr_err': [],
        'dist': [],
        'num_gaussians': [],
        'detection_name': []
    }
    
    target_frame_set = set(target_frames)
    
    for i, frame in enumerate(initial_data['frames']):
        if frame in target_frame_set:
            filtered_data['frames'].append(frame)
            filtered_data['trans_err'].append(initial_data['trans_err'][i])
            filtered_data['vel_err'].append(initial_data['vel_err'][i])
            filtered_data['scale_err'].append(initial_data['scale_err'][i])
            filtered_data['orient_err'].append(initial_data['orient_err'][i])
            filtered_data['attr_err'].append(initial_data['attr_err'][i])
            filtered_data['dist'].append(initial_data['dist'][i])
            filtered_data['num_gaussians'].append(initial_data['num_gaussians'][i])
            filtered_data['detection_name'].append(initial_data['detection_name'][i])

    return filtered_data

def plot_frame_vs_error(plot_data: Dict[str, List], output_dir: str, group_size: int = 10, initial_plot_data: Optional[Dict[str, List]] = None):
    """프레임 인덱스 vs 오차 scatter plot과 추세선을 생성합니다."""
    error_types = ['trans_err', 'scale_err', 'orient_err']
    error_names = ['Translation Error', 'Scale Error', 'Orientation Error']
    
    # 각 축별 최대값 계산
    max_values = calculate_max_values_for_axes(plot_data, initial_plot_data)
    print(f"축별 최대값: {max_values}")
    
    # 각 오차 타입별로 개별 그래프 생성
    for error_type, error_name in zip(error_types, error_names):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 전체 데이터를 scatter plot으로 표시
        frames = plot_data['frames']
        errors = plot_data[error_type]
        
        # 현재 데이터 scatter plot
        ax.scatter(frames, errors, alpha=0.6, s=30, c='blue', label='Current Data')
        
        # 현재 데이터 추세선
        if len(frames) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(frames, errors)
            x_trend = np.linspace(0, max_values['frames'], 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'b--', alpha=0.8, linewidth=2, 
                   label=f'Current Trend')
        
        # 초기값 데이터가 있으면 함께 표시
        if initial_plot_data:
            initial_frames = initial_plot_data['frames']
            initial_errors = initial_plot_data[error_type]
            ax.scatter(initial_frames, initial_errors, alpha=0.4, s=20, c='red', 
                      marker='x', label='Initial Data')
            
            # 초기값 데이터 추세선
            if len(initial_frames) > 1:
                slope_init, intercept_init, r_value_init, p_value_init, std_err_init = stats.linregress(initial_frames, initial_errors)
                x_trend_init = np.linspace(0, max_values['frames'], 100)
                y_trend_init = slope_init * x_trend_init + intercept_init
                ax.plot(x_trend_init, y_trend_init, 'r:', alpha=0.8, linewidth=2, 
                       label=f'Initial Trend')
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(error_name)
        title_suffix = "with Initial Comparison" if initial_plot_data else ""
        ax.set_title(f'{error_name} vs Frame Index - Scatter Plot {title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # y축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_ylim(0, max_values[error_type] * 1.05)
        
        # x축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_xlim(0, max_values['frames'] * 1.05)
        
        plt.tight_layout()
        # 파일 저장
        plt.savefig(os.path.join(output_dir, f'frame_vs_{error_type}_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_distance_vs_error(plot_data: Dict[str, List], output_dir: str, group_size: int = 10, initial_plot_data: Optional[Dict[str, List]] = None):
    """거리 vs 오차 scatter plot과 추세선을 생성합니다."""
    error_types = ['trans_err', 'scale_err', 'orient_err']
    error_names = ['Translation Error', 'Scale Error', 'Orientation Error']
    
    # 각 축별 최대값 계산
    max_values = calculate_max_values_for_axes(plot_data, initial_plot_data)
    
    # 각 오차 타입별로 개별 그래프 생성
    for error_type, error_name in zip(error_types, error_names):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 전체 데이터를 scatter plot으로 표시
        distances = plot_data['dist']
        errors = plot_data[error_type]
        
                # 현재 데이터 scatter plot
        ax.scatter(distances, errors, alpha=0.6, s=30, c='blue', label='Current Data')
        
        # 현재 데이터 추세선
        if len(distances) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(distances, errors)
            x_trend = np.linspace(0, max_values['dist'], 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'b--', alpha=0.8, linewidth=2, 
                   label=f'Current Trend')
        
        # 초기값 데이터가 있으면 함께 표시
        if initial_plot_data:
            initial_distances = initial_plot_data['dist']
            initial_errors = initial_plot_data[error_type]
            ax.scatter(initial_distances, initial_errors, alpha=0.4, s=20, c='red', 
                      marker='x', label='Initial Data')
            
            # 초기값 데이터 추세선
            if len(initial_distances) > 1:
                slope_init, intercept_init, r_value_init, p_value_init, std_err_init = stats.linregress(initial_distances, initial_errors)
                x_trend_init = np.linspace(0, max_values['dist'], 100)
                y_trend_init = slope_init * x_trend_init + intercept_init
                ax.plot(x_trend_init, y_trend_init, 'r:', alpha=0.8, linewidth=2, 
                       label=f'Initial Trend')
        
        ax.set_xlabel('Distance')
        ax.set_ylabel(error_name)
        title_suffix = "with Initial Comparison" if initial_plot_data else ""
        ax.set_title(f'{error_name} vs Distance - Scatter Plot {title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # y축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_ylim(0, max_values[error_type] * 1.05)
        
        # x축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_xlim(0, max_values['dist'] * 1.05)
        
        plt.tight_layout()
        # 파일 저장
        plt.savefig(os.path.join(output_dir, f'distance_vs_{error_type}_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_gaussians_vs_error(plot_data: Dict[str, List], output_dir: str, group_size: int = 10, initial_plot_data: Optional[Dict[str, List]] = None):
    """가우시안 개수 vs 오차 scatter plot과 추세선을 생성합니다."""
    error_types = ['trans_err', 'scale_err', 'orient_err']
    error_names = ['Translation Error', 'Scale Error', 'Orientation Error']
    
    # 각 축별 최대값 계산
    max_values = calculate_max_values_for_axes(plot_data, initial_plot_data)
    
    # 각 오차 타입별로 개별 그래프 생성
    for error_type, error_name in zip(error_types, error_names):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 전체 데이터를 scatter plot으로 표시
        num_gaussians = plot_data['num_gaussians']
        errors = plot_data[error_type]
        
                # 현재 데이터 scatter plot
        ax.scatter(num_gaussians, errors, alpha=0.6, s=30, c='blue', label='Current Data')
        
        # 현재 데이터 추세선
        if len(num_gaussians) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(num_gaussians, errors)
            x_trend = np.linspace(0, max_values['num_gaussians'], 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'b--', alpha=0.8, linewidth=2, 
                   label=f'Current Trend')         
        
        ax.set_xlabel('Number of Gaussians')
        ax.set_ylabel(error_name)
        title_suffix = "with Initial Comparison" if initial_plot_data else ""
        ax.set_title(f'{error_name} vs Number of Gaussians - Scatter Plot {title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # y축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_ylim(0, max_values[error_type] * 1.05)
        
        # x축 범위를 최대값으로 고정 (약간의 여백 추가)
        ax.set_xlim(0, max_values['num_gaussians'] * 1.05)
        
        plt.tight_layout()
        # 파일 저장
        plt.savefig(os.path.join(output_dir, f'gaussians_vs_{error_type}_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_frame_count_vs_mae(plot_data: Dict[str, List], output_dir: str, initial_plot_data: Optional[Dict[str, List]] = None):
    """프레임 개수 vs MAE scatter plot을 생성합니다."""
    error_types = ['trans_err', 'scale_err', 'orient_err']
    error_names = ['Translation Error', 'Scale Error', 'Orientation Error']
    
    # 각 오차 타입별로 개별 그래프 생성
    for error_type, error_name in zip(error_types, error_names):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 현재 데이터에서 인스턴스별 프레임 개수와 MAE 계산
        unique_instances = list(set(plot_data['instance_ids']))
        frame_counts = []
        mae_values = []
        
        for instance_id in unique_instances:
            instance_mask = [inst_id == instance_id for inst_id in plot_data['instance_ids']]
            instance_errors = [e for e, mask in zip(plot_data[error_type], instance_mask) if mask]
            
            if instance_errors:
                frame_counts.append(len(instance_errors))
                mae_values.append(np.mean(np.abs(instance_errors)))
        
        # 현재 데이터 scatter plot
        ax.scatter(frame_counts, mae_values, alpha=0.7, s=50, c='blue', label='Current Data')
        
        # 현재 데이터 추세선
        if len(frame_counts) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(frame_counts, mae_values)
            x_trend = np.linspace(min(frame_counts), max(frame_counts), 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'b--', alpha=0.8, linewidth=2, 
                   label=f'Current Trend')
        
        # 초기값 데이터가 있으면 함께 표시
        if initial_plot_data:
            initial_unique_instances = list(set(initial_plot_data['instance_ids']))
            initial_frame_counts = []
            initial_mae_values = []
            
            for instance_id in initial_unique_instances:
                instance_mask = [inst_id == instance_id for inst_id in initial_plot_data['instance_ids']]
                instance_errors = [e for e, mask in zip(initial_plot_data[error_type], instance_mask) if mask]
                
                if instance_errors:
                    initial_frame_counts.append(len(instance_errors))
                    initial_mae_values.append(np.mean(np.abs(instance_errors)))
            
            ax.scatter(initial_frame_counts, initial_mae_values, alpha=0.5, s=40, c='red', 
                      marker='x', label='Initial Data')
            
            # 초기값 데이터 추세선
            if len(initial_frame_counts) > 1:
                slope_init, intercept_init, r_value_init, p_value_init, std_err_init = stats.linregress(initial_frame_counts, initial_mae_values)
                x_trend_init = np.linspace(min(initial_frame_counts), max(initial_frame_counts), 100)
                y_trend_init = slope_init * x_trend_init + intercept_init
                ax.plot(x_trend_init, y_trend_init, 'r:', alpha=0.8, linewidth=2, 
                       label=f'Initial Trend')
        
        ax.set_xlabel('Number of Frames per Instance')
        ax.set_ylabel(f'{error_name} MAE')
        title_suffix = "with Initial Comparison" if initial_plot_data else ""
        ax.set_title(f'{error_name} MAE vs Frame Count per Instance {title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        # 파일 저장
        plt.savefig(os.path.join(output_dir, f'frame_count_vs_{error_type}_mae.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_statistics(plot_data: Dict[str, List], output_dir: str):
    """요약 통계를 생성하고 저장합니다."""
    error_types = ['trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err']
    
    summary_stats = {}
    unique_instances = list(set(plot_data['instance_ids']))
    
    for instance_id in unique_instances:
        instance_mask = [inst_id == instance_id for inst_id in plot_data['instance_ids']]
        instance_stats = {'instance_id': instance_id}
        
        # 각 오차 타입별 통계
        for error_type in error_types:
            instance_errors = [e for e, mask in zip(plot_data[error_type], instance_mask) if mask]
            instance_stats[f'{error_type}_mean'] = np.mean(instance_errors)
            instance_stats[f'{error_type}_std'] = np.std(instance_errors)
            instance_stats[f'{error_type}_min'] = np.min(instance_errors)
            instance_stats[f'{error_type}_max'] = np.max(instance_errors)
        
        # 가우시안 개수 (인스턴스별로 고정값)
        instance_gaussians = [g for g, mask in zip(plot_data['num_gaussians'], instance_mask) if mask]
        instance_stats['num_gaussians'] = instance_gaussians[0] if instance_gaussians else 0
        
        summary_stats[instance_id] = instance_stats
    
    # JSON으로 저장
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    print(f"요약 통계가 {os.path.join(output_dir, 'summary_statistics.json')}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='인스턴스별 오차 데이터 시각화')
    parser.add_argument('--input_json', type=str, required=False,
                    default='/workspace/drivestudio/output/box_experiments_0826/rotstartfrom_try7_p2d15000_rotfull/box_poses/instance_wise_frame_err_data.json',
                       help='입력 JSON 파일 경로')
    parser.add_argument('--initial_json', type=str, required=False,
                       default='/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini/001/instances/instance_wise_frame_err_data.json',
                       help='초기값 오차 JSON 파일 경로 (비교용)')
    parser.add_argument('--output_dir', type=str, required=False,
                       default=None,
                       help='출력 디렉토리 경로')
    parser.add_argument('--group_size', type=int, required=False,
                       default=0,
                       help='인스턴스를 묶을 그룹 크기 (기본값: 5, 0이면 모든 인스턴스를 하나의 그래프에 표시)')
    
    args = parser.parse_args()
    # target_node_class = ["RigidNodes", "SMPLNodes", "DeformableNodes"]
    target_node_class = ["RigidNodes"]

    # 출력 디렉토리 생성
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_json) + '/plots'
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"JSON 파일 로딩: {args.input_json}")
    data = load_json_data(args.input_json)
    
    print("데이터 추출 중...")
    plot_data = extract_data_for_plotting(data, target_node_class)
    print(f"입력 데이터 총 {len(set(plot_data['instance_ids']))}개 인스턴스, {len(plot_data['frames'])}개 데이터 포인트")

    # 초기값 데이터 로드 (있는 경우)
    initial_plot_data = None
    if args.initial_json:
        initial_plot_data = load_and_process_initial_data(args.initial_json, target_node_class)
        if initial_plot_data:
            print(f"초기값 데이터 총 {len(set(initial_plot_data['instance_ids']))}개 인스턴스, {len(initial_plot_data['frames'])}개 데이터 포인트")
    
    
    print("프레임 vs 오차 scatter plot 생성 중...")
    plot_frame_vs_error(plot_data, output_dir, args.group_size, initial_plot_data)
    
    print("거리 vs 오차 scatter plot 생성 중...")
    plot_distance_vs_error(plot_data, output_dir, args.group_size, initial_plot_data)
    
    print("가우시안 개수 vs 오차 scatter plot 생성 중...")
    plot_gaussians_vs_error(plot_data, output_dir, args.group_size, initial_plot_data)
    
    print("프레임 개수 vs MAE scatter plot 생성 중...")
    plot_frame_count_vs_mae(plot_data, output_dir, initial_plot_data)
    
    print("요약 통계 생성 중...")
    create_summary_statistics(plot_data, output_dir)
    
    # MAE 정보 저장
    print("MAE 정보 저장 중...")
    instance_mae_list = calculate_mae_per_instance(plot_data)
    mae_info = {
        'instances_by_mae_rank': [
            {'rank': i+1, 'instance_id': instance_id, 'trans_err_mae': mae}
            for i, (instance_id, mae) in enumerate(instance_mae_list)
        ]
    }
    with open(os.path.join(output_dir, 'mae_ranking.json'), 'w', encoding='utf-8') as f:
        json.dump(mae_info, f, indent=2, ensure_ascii=False)
    
    print(f"모든 그래프가 {output_dir}에 저장되었습니다.")
    print("생성된 파일:")
    print("  - frame_vs_[error_type]_scatter.png: 프레임 인덱스 vs 각 오차 타입 (scatter plot + 추세선)")
    print("  - distance_vs_[error_type]_scatter.png: 거리 vs 각 오차 타입 (scatter plot + 추세선)") 
    print("  - gaussians_vs_[error_type]_scatter.png: 가우시안 개수 vs 각 오차 타입 (scatter plot + 추세선)")
    print("  - frame_count_vs_[error_type]_mae.png: 인스턴스별 프레임 개수 vs MAE (scatter plot + 추세선)")
    print("  - summary_statistics.json: 요약 통계")
    print("  - mae_ranking.json: 인스턴스별 trans_err MAE 순위")
    print("  (각 오차 타입: trans_err, scale_err, orient_err)")
    print("  (모든 그래프가 scatter plot으로 표시됨)")
    if initial_plot_data:
        print("  (초기값과 현재값 각각에 대한 추세선이 표시됨: 파란색 점선=현재값, 빨간색 점선=초기값)")

if __name__ == "__main__":
    main()
