import argparse
import numpy as np
import tqdm
from typing import Callable, Tuple, List, Dict, Optional
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyquaternion import Quaternion
import open3d as o3d
import open3d.visualization
import ctypes

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionBox, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
import abc
from typing import Union

class EvalBox(abc.ABC):
    """ Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 ego_rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 num_pts: int = -1):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.

        # Assert data for shape and NaNs.
        assert type(sample_token) == str, 'Error: sample_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 4, 'Error: Rotation must have 4 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Velocity must have 2 elements!'

        assert len(ego_translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(ego_translation)), 'Error: Translation may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.ego_rotation = ego_rotation
        self.num_pts = num_pts

    @property
    def ego_dist(self) -> float:
        """ Compute the distance from this box to the ego vehicle in 2D. """
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        pass


def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> Tuple[EvalBoxes, Dict[str, List]]:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()
    all_ann_tokens = defaultdict(list)
    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        sample_ann_tokens = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                )
                sample_ann_tokens.append(sample_annotation_token)
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
                sample_ann_tokens.append(sample_annotation_token)
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)
        all_ann_tokens[sample_token].extend(sample_ann_tokens)
    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations, all_ann_tokens

def add_ego_pose(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the ego pose information (translation and rotation) to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with ego pose information.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Get ego pose transformation
        ego_translation_global = np.array(pose_record['translation'])
        ego_rotation_global = Quaternion(pose_record['rotation'])

        for box in eval_boxes[sample_token]:
            # Convert global coordinates to ego vehicle local coordinates
            
            # Step 1: Get relative position vector in global coordinates
            box_translation_global = np.array(box.translation)
            relative_translation_global = box_translation_global - ego_translation_global
            
            # Step 2: Rotate this position vector to ego vehicle's local coordinate system
            # This transforms "relative position in global frame" to "relative position in ego frame"
            # Example: if ego is rotated 90°, what's "north" in global becomes "left" in ego frame
            ego_translation_array = ego_rotation_global.inverse.rotate(relative_translation_global)
            
            # Step 3: Transform box rotation to ego coordinates
            box_rotation_global = Quaternion(list(box.rotation))  # type: ignore
            ego_rotation = ego_rotation_global.inverse * box_rotation_global
            
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = tuple(ego_translation_array)
                # Add ego_rotation attribute dynamically
                setattr(box, 'ego_rotation', tuple([ego_rotation.w, ego_rotation.x, ego_rotation.y, ego_rotation.z]))  # type: ignore
            else:
                raise NotImplementedError

    return eval_boxes

def get_box_corners(translation, size, rotation):
    """3D 박스의 8개 꼭짓점을 계산합니다.
    
    Args:
        translation: [x, y, z] 중심점
        size: [width, length, height] 크기
        rotation: [w, x, y, z] 쿼터니언
        
    Returns:
        8x3 numpy array: 8개 꼭짓점의 좌표
    """
    w, l, h = size
    
    # 로컬 좌표계에서의 8개 꼭짓점 (중심이 원점)
    corners_local = np.array([
        [-l/2, -w/2, -h/2],  # 0: 좌하후
        [l/2, -w/2, -h/2],   # 1: 우하후
        [l/2, w/2, -h/2],    # 2: 우상후
        [-l/2, w/2, -h/2],   # 3: 좌상후
        [-l/2, -w/2, h/2],   # 4: 좌하전
        [l/2, -w/2, h/2],    # 5: 우하전
        [l/2, w/2, h/2],     # 6: 우상전
        [-l/2, w/2, h/2]     # 7: 좌상전
    ])
    
    # pyquaternion을 사용하여 회전 적용
    q = Quaternion(rotation)  # [w, x, y, z] 순서
    rotation_matrix = q.rotation_matrix
    # 올바른 회전 변환: (rotation_matrix @ corners.T).T
    corners_rotated = (rotation_matrix @ corners_local.T).T
    
    # 평행이동 적용
    corners_world = corners_rotated + np.array(translation)
    
    return corners_world

def draw_3d_box(ax, corners, color='blue', alpha=0.3, edge_color='black'):
    """3D 박스를 그립니다.
    
    Args:
        ax: matplotlib 3D axis
        corners: 8x3 numpy array, 박스의 8개 꼭짓점
        color: 박스 면의 색상
        alpha: 투명도
        edge_color: 테두리 색상
    """
    # 12개의 면을 정의 (각 면은 4개의 꼭짓점으로 구성)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # 아래면
        [corners[4], corners[5], corners[6], corners[7]],  # 위면
        [corners[0], corners[1], corners[5], corners[4]],  # 앞면
        [corners[2], corners[3], corners[7], corners[6]],  # 뒷면
        [corners[1], corners[2], corners[6], corners[5]],  # 오른쪽면
        [corners[4], corners[7], corners[3], corners[0]]   # 왼쪽면
    ]
    
    # Poly3DCollection을 사용해서 면들을 그리기
    poly3d = [[tuple(face[j]) for j in range(len(face))] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, 
                                        facecolors=color, 
                                        linewidths=1, 
                                        edgecolors=edge_color,
                                        alpha=alpha))

# ===========================
# Open3D 시각화 유틸리티
# ===========================

# Open3D LineSet 생성을 위한 에지 인덱스 (12개)
OPEN3D_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 아래면
    [4, 5], [5, 6], [6, 7], [7, 4],  # 위면
    [0, 4], [1, 5], [2, 6], [3, 7]   # 옆면
]

# -----------------------------
# NEW: Helper for front center sphere
# -----------------------------

def create_open3d_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """지정한 중심과 색상의 구(Sphere) Mesh를 생성합니다.

    Args:
        center (np.ndarray): 중심 좌표 (3,)
        radius (float): 구의 반지름
        color (Tuple[float, float, float]): RGB 컬러 (0~1)

    Returns:
        o3d.geometry.TriangleMesh: 시각화용 Sphere Mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def create_open3d_box(corners: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """8개 꼭짓점 정보로부터 Open3D 두꺼운 선으로 이루어진 육면체 객체를 생성합니다.

    Args:
        corners (np.ndarray): (8, 3) 형태의 꼭짓점 좌표
        color (Tuple[float, float, float]): RGB 컬러 (0~1)

    Returns:
        o3d.geometry.TriangleMesh: 시각화용 두꺼운 선 박스
    """
    # 선 두께 설정
    line_radius = 0.05  # 선의 반지름 (두께 조절)
    
    # 모든 실린더를 합칠 메쉬
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # 12개의 모서리에 대해 실린더 생성
    for line_indices in OPEN3D_BOX_LINES:
        start_point = corners[line_indices[0]]
        end_point = corners[line_indices[1]]
        
        # 두 점 사이의 거리 계산
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 1e-6:  # 너무 짧은 선은 건너뛰기
            continue
            
        # 실린더 생성 (Z축 방향으로 생성됨)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=line_radius, 
            height=line_length,
            resolution=8  # 실린더의 해상도 (낮으면 성능 향상)
        )
        
        # 실린더를 올바른 방향으로 회전시키기
        # Z축 단위벡터
        z_axis = np.array([0, 0, 1])
        # 선의 방향 벡터
        line_direction = line_vector / line_length
        
        # 회전축 계산 (외적)
        rotation_axis = np.cross(z_axis, line_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:  # 평행하지 않은 경우
            # 회전각 계산
            cos_angle = np.dot(z_axis, line_direction)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # 회전축 정규화
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # 회전 행렬 생성
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle
            )
            
            # 실린더 회전
            cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        
        # 실린더를 시작점으로 이동 (실린더 중심이 선의 중점이 되도록)
        cylinder_center = (start_point + end_point) / 2
        cylinder.translate(cylinder_center)
        
        # 색상 적용
        cylinder.paint_uniform_color(color)
        
        # 메쉬 합치기
        combined_mesh += cylinder
    
    return combined_mesh


def load_lidar_pointcloud(nusc: NuScenes, sample_token: str) -> Optional[np.ndarray]:
    """NuScenes sample에서 LiDAR 포인트 클라우드를 로드하고 ego vehicle 좌표계로 변환합니다.
    
    Args:
        nusc: NuScenes 객체
        sample_token: sample token
        
    Returns:
        ego vehicle 좌표계의 포인트 클라우드 numpy array (N, 3) 또는 None
    """
    try:
        # sample 정보 가져오기
        sample = nusc.get('sample', sample_token)
        
        # LiDAR 데이터 토큰 가져오기
        lidar_token = sample['data']['LIDAR_TOP']
        
        # sample_data 정보 가져오기
        sample_data = nusc.get('sample_data', lidar_token)
        
        # LiDAR 파일 경로
        lidar_path = os.path.join(nusc.dataroot, sample_data['filename'])
        
        if not os.path.exists(lidar_path):
            print(f"⚠️ LiDAR 파일을 찾을 수 없습니다: {lidar_path}")
            return None
            
        # LiDAR 포인트 클라우드 로드 (센서 좌표계)
        pc = LidarPointCloud.from_file(lidar_path)
        
        # LiDAR extrinsic calibration 정보 가져오기
        calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        
        # LiDAR extrinsic: 센서 -> ego vehicle 변환
        lidar_translation = np.array(calibrated_sensor['translation'])
        lidar_rotation = Quaternion(calibrated_sensor['rotation'])
        
        # 포인트를 ego vehicle 좌표계로 변환
        # Step 1: LiDAR 센서 좌표계의 포인트들 (x, y, z)
        points_sensor = pc.points[:3, :]  # (3, N)
        
        # Step 2: LiDAR rotation 적용
        points_rotated = lidar_rotation.rotation_matrix @ points_sensor  # (3, N)
        
        # Step 3: LiDAR translation 적용
        points_ego = points_rotated + lidar_translation.reshape(3, 1)  # (3, N)
        
        # (N, 3) 형태로 변환하여 반환
        return points_ego.T
        
    except Exception as e:
        print(f"⚠️ LiDAR 데이터 로드 실패: {e}")
        return None


def create_open3d_pointcloud(points: np.ndarray, color: Optional[Tuple[float, float, float]] = None,
                            max_points: int = 50000) -> o3d.geometry.PointCloud:
    """numpy array로부터 Open3D PointCloud 객체를 생성합니다.
    
    Args:
        points: (N, 3) 형태의 포인트 좌표
        color: RGB 색상 (0~1), None이면 거리에 따른 색상 사용
        max_points: 최대 포인트 개수 (성능을 위해 제한)
        
    Returns:
        Open3D PointCloud 객체
    """
    # 포인트 개수 제한
    if len(points) > max_points:
        # 랜덤 샘플링
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    # Open3D PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if color is not None:
        # 단일 색상 적용
        pcd.paint_uniform_color(color)
    else:
        # 거리에 따른 색상 (회색조)
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.percentile(distances, 95)  # 95 percentile로 스케일링
        normalized_distances = np.clip(distances / max_dist, 0, 1)
        
        # 거리가 가까우면 밝은 회색, 멀면 어두운 회색
        colors = np.column_stack([1 - normalized_distances * 0.7] * 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


# noinspection PyBroadException
def visualize_ego_translations_open3d(gaussian_boxes: Optional[EvalBoxes] = None, 
                                     pred_boxes: Optional[EvalBoxes] = None, 
                                     gt_boxes: Optional[EvalBoxes] = None, 
                                     scene_name: Optional[str] = None,
                                     score_threshold: Optional[float] = None, 
                                     save_path: Optional[str] = None, 
                                     max_boxes: int = -1,
                                     sample_token: Optional[str] = None,
                                     show_lidar: bool = False,
                                     nusc: Optional[NuScenes] = None,
                                     max_lidar_points: int = 50000) -> None:
    """Open3D를 이용하여 gaussian_boxes, pred_boxes, gt_boxes를 3D로 시각화합니다.

    Args:
        gaussian_boxes: Gaussian 박스들 (빨간색)
        pred_boxes: 예측 박스들 (파란색)
        gt_boxes: Ground truth 박스들 (초록색)
        scene_name: (선택) scene 이름 (제목 표시용)
        score_threshold: (선택) score threshold (제목 표시용)
        save_path: (선택) 저장할 파일 경로
        max_boxes: 최대 박스 개수
        sample_token: (선택) 특정 sample만 시각화
        show_lidar: LiDAR 포인트 클라우드 표시 여부
        nusc: NuScenes 객체 (LiDAR 데이터 로딩에 필요)
        max_lidar_points: 최대 LiDAR 포인트 개수
    """

    geometries = []

    # Coordinate frame 추가
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))

    gaussian_count, pred_count, gt_count = 0, 0, 0
    center_translation = None

    # sample_token이 지정된 경우 해당 sample의 박스들만 사용
    sample_tokens_to_process = []
    
    # 처리할 sample_tokens 결정
    if sample_token:
        # 특정 sample만 처리
        for boxes in [gaussian_boxes, pred_boxes, gt_boxes]:
            if boxes is not None and sample_token in boxes.sample_tokens:
                sample_tokens_to_process = [sample_token]
                break
        if not sample_tokens_to_process:
            print(f"⚠️ Sample token '{sample_token}'을 찾을 수 없습니다.")
            return
    else:
        # 모든 sample 처리 (기존 동작)
        all_sample_tokens = set()
        for boxes in [gaussian_boxes, pred_boxes, gt_boxes]:
            if boxes is not None:
                all_sample_tokens.update(boxes.sample_tokens)
        sample_tokens_to_process = list(all_sample_tokens)

    # 첫 번째 박스의 translation을 center로 설정
    for sample_token_iter in sample_tokens_to_process:
        for boxes in [gaussian_boxes, pred_boxes, gt_boxes]:
            if boxes is not None and sample_token_iter in boxes.sample_tokens:
                for box in boxes[sample_token_iter]:
                    if (hasattr(box, 'translation') and box.translation is not None and
                            hasattr(box, 'size') and box.size is not None and
                            hasattr(box, 'rotation') and box.rotation is not None):
                        center_translation = np.array(box.translation)
                        break
                if center_translation is not None:
                    break
            if center_translation is not None:
                break
        if center_translation is not None:
            break

    if center_translation is None:
        print("기준이 될 박스를 찾을 수 없습니다.")
        return

    # LiDAR 포인트 클라우드 추가 (특정 sample이 지정된 경우에만)
    if show_lidar and nusc is not None and sample_token is not None:
        print(f"🔍 LiDAR 포인트 클라우드 로딩 중... (sample: {sample_token[:8]}...)")
        lidar_points = load_lidar_pointcloud(nusc, sample_token)
        if lidar_points is not None:
            print(f"✅ {len(lidar_points):,}개의 LiDAR 포인트를 로드했습니다.")  
            # Open3D PointCloud 생성 및 추가
            lidar_pcd = create_open3d_pointcloud(lidar_points, 
                                               color=(0.5, 0.5, 0.5),  # 회색
                                               max_points=max_lidar_points)
            geometries.append(lidar_pcd)
        else:
            print("⚠️ LiDAR 포인트 클라우드를 로드할 수 없습니다.")
    elif show_lidar and sample_token is None:
        print("⚠️ LiDAR 시각화는 특정 sample이 지정된 경우에만 가능합니다. --sample_token을 사용하세요.")

    # 특정 sample 시각화 시 ego 좌표계 사용, 전체 시각화 시 global 좌표계 사용
    use_ego_coordinates = sample_token is not None and len(sample_tokens_to_process) == 1

    # 처리할 sample들에 대해서만 박스 추가
    for sample_token_iter in sample_tokens_to_process:
        # Gaussian boxes (Red)
        if gaussian_boxes is not None and sample_token_iter in gaussian_boxes.sample_tokens:
            for box in gaussian_boxes[sample_token_iter]:
                if max_boxes > 0 and gaussian_count >= max_boxes:
                    break  # 개수 제한 도달
                if (hasattr(box, 'translation') and box.translation is not None and
                        hasattr(box, 'size') and box.size is not None and
                        hasattr(box, 'rotation') and box.rotation is not None):
                    
                    # ego 좌표계 사용 여부에 따라 좌표 선택
                    if use_ego_coordinates and hasattr(box, 'ego_translation') and hasattr(box, 'ego_rotation'):
                        translation = box.ego_translation  # type: ignore
                        rotation = getattr(box, 'ego_rotation')  # type: ignore
                        relative_translation = np.array(translation)
                    else:
                        # center 기준으로 상대 위치 계산 (기존 방식)
                        relative_translation = np.array(box.translation) - center_translation
                        rotation = box.rotation
                    
                    corners = get_box_corners(relative_translation, box.size, rotation)
                    geometries.append(create_open3d_box(corners, (1.0, 0.0, 0.0)))  # Red

                    # 앞면 중심점 시각화 (Red)
                    front_center = (corners[1] + corners[6]) / 2 
                    geometries.append(create_open3d_sphere(front_center, radius=0.1, color=(1.0, 0.0, 0.0)))

                    gaussian_count += 1
            if max_boxes > 0 and gaussian_count >= max_boxes:
                break

        # Prediction boxes (Blue)
        if pred_boxes is not None and sample_token_iter in pred_boxes.sample_tokens:
            for box in pred_boxes[sample_token_iter]:
                if max_boxes > 0 and pred_count >= max_boxes:
                    break  # 개수 제한 도달
                if (hasattr(box, 'translation') and box.translation is not None and
                        hasattr(box, 'size') and box.size is not None and
                        hasattr(box, 'rotation') and box.rotation is not None):
                    
                    # ego 좌표계 사용 여부에 따라 좌표 선택
                    if use_ego_coordinates and hasattr(box, 'ego_translation') and hasattr(box, 'ego_rotation'):
                        translation = box.ego_translation  # type: ignore
                        rotation = getattr(box, 'ego_rotation')  # type: ignore
                        relative_translation = np.array(translation)
                    else:
                        # center 기준으로 상대 위치 계산 (기존 방식)
                        relative_translation = np.array(box.translation) - center_translation
                        rotation = box.rotation
                    
                    corners = get_box_corners(relative_translation, box.size, rotation)
                    geometries.append(create_open3d_box(corners, (0.0, 0.0, 1.0)))  # Blue

                    # 앞면 중심점 시각화 (Blue)
                    front_center = (corners[1] + corners[6]) / 2 
                    geometries.append(create_open3d_sphere(front_center, radius=0.1, color=(0.0, 0.0, 1.0)))

                    pred_count += 1
            if max_boxes > 0 and pred_count >= max_boxes:
                break

        # Ground truth boxes (black)
        if gt_boxes is not None and sample_token_iter in gt_boxes.sample_tokens:
            for box in gt_boxes[sample_token_iter]:
                if max_boxes > 0 and gt_count >= max_boxes:
                    break  # 개수 제한 도달
                if (hasattr(box, 'translation') and box.translation is not None and
                        hasattr(box, 'size') and box.size is not None and
                        hasattr(box, 'rotation') and box.rotation is not None):

                    # ego 좌표계 사용 여부에 따라 좌표 선택
                    if use_ego_coordinates and hasattr(box, 'ego_translation') and hasattr(box, 'ego_rotation'):
                        translation = box.ego_translation  # type: ignore
                        rotation = getattr(box, 'ego_rotation')  # type: ignore
                        relative_translation = np.array(translation)
                    else:
                        # center 기준으로 상대 위치 계산 (기존 방식)
                        relative_translation = np.array(box.translation) - center_translation
                        rotation = box.rotation
                    
                    corners = get_box_corners(relative_translation, box.size, rotation)
                    geometries.append(create_open3d_box(corners, (0.0, 0.0, 0.0)))  # Green

                    # 앞면 중심점 시각화 (Green)
                    front_center = (corners[1] + corners[6]) / 2 
                    geometries.append(create_open3d_sphere(front_center, radius=0.1, color=(0.0, 0.0, 0.0)))

                    gt_count += 1
            if max_boxes > 0 and gt_count >= max_boxes:
                break

        if max_boxes > 0 and (gaussian_count + pred_count + gt_count) >= max_boxes:
            break

    if gaussian_count == 0 and pred_count == 0 and gt_count == 0:
        print("시각화할 박스가 없습니다.")
        return

    # 윈도우 이름 설정
    window_name = "Open3D Visualization of Detection Boxes"
    subtitle_parts = []
    if scene_name:
        subtitle_parts.append(f"Scene: {scene_name}")
    if sample_token:
        subtitle_parts.append(f"Sample: {sample_token[:8]}...")
    if score_threshold is not None and score_threshold > 0:
        subtitle_parts.append(f"Score≥{score_threshold}")
    subtitle_parts.append(f"Gaussian: {gaussian_count}, Pred: {pred_count}, GT: {gt_count}")
    if subtitle_parts:
        window_name += " (" + " | ".join(subtitle_parts) + ")"

    # ---------------------------
    # 시각화 (온스크린 or 오프스크린)
    # ---------------------------

    # 1) 오프스크린 렌더링 모드가 필요한 경우 (save_path 지정 or GUI 사용 불가)
    if save_path is not None:
        try:
            # 저장 경로 디렉토리 생성
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"📁 디렉토리 생성: {save_dir}")
            
            print(f"🎨 오프스크린 렌더링 시작... ({len(geometries)}개 객체)")
            success = False 
            
            vis = o3d.visualization.Visualizer()  # type: ignore
            # vis.create_window(visible=False, width=1920, height=1080)
            vis.create_window(visible=False, width=3840, height=2160)
            
            # 렌더링 옵션 설정
            render_option = vis.get_render_option()
            render_option.background_color = np.array([1, 1, 1])
            render_option.point_size = 6.0
            # render_option.line_width = 8.0  # 박스 선을 더 두껍게 설정
            
            # 기하학적 객체들 추가
            for g in geometries:
                vis.add_geometry(g)
            
            # --------------------------------------
            # 카메라 시점: Top-Down(조감) 뷰로 변경
            #   • front : (0, 0, -1)  → 위에서 아래로 내려다봄
            #   • up    : (0, -1, 0) → Y-축을 화면 위쪽으로 지정
            # --------------------------------------
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])   # 카메라가 -Z 방향(아래)으로 바라보도록 설정
            ctr.set_up([0, -1, 0])      # 화면의 위쪽을 -Y 방향으로 맞춤 (XY 평면 기준)
            ctr.set_lookat([0, 0, 0])   # 원점(센서 위치)을 바라보도록 설정
            
            # --------------------------------------
            # 직교 투영(Orthographic Projection) 활성화
            #   → 원근법 없이 모든 객체가 동일한 스케일로 보이도록 설정
            # --------------------------------------
            # Open3D에서 직교 투영을 위해 field of view를 매우 작게 설정
            # 이렇게 하면 원근법 효과가 거의 사라져서 직교 투영과 유사한 효과를 얻을 수 있음
            ctr.change_field_of_view(step=-500)  # FOV를 매우 작게 설정하여 직교 투영 효과
            ctr.set_zoom(0.15)  # 적절한 줌 레벨 설정

            
            # 여러 번 렌더링하여 안정화
            for _ in range(3):
                vis.poll_events()
                vis.update_renderer()
            
            # Float buffer로 이미지 캡처 (더 안정적)
            try:
                image = vis.capture_screen_float_buffer(do_render=True)
                image_np = np.asarray(image)
                
                # Float buffer를 0-255 범위로 변환
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # PIL로 이미지 저장
                from PIL import Image
                pil_image = Image.fromarray(image_np)
                pil_image.save(save_path)
                success = True
                
            except Exception as e2:
                print(f"⚠️ Float buffer 방법 실패: {e2}")
                # 최후의 수단: 기본 capture_screen_image
                success = vis.capture_screen_image(save_path)
            
            vis.destroy_window()
            
            # 결과 확인 및 피드백
            if success and os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 1000:  # 최소 1KB 이상이어야 유효한 이미지
                    print(f"✅ 3D 시각화 결과 저장 완료:")
                    print(f"   📂 경로: {save_path}")
                    print(f"   📊 크기: {file_size:,} bytes")
                else:
                    print(f"⚠️ 이미지 파일이 너무 작습니다: {file_size} bytes")
            else:
                print(f"❌ 이미지 저장 실패: {save_path}")
                
        except Exception as e:
            print(f"⚠️ 전체 렌더링 과정 실패: {e}")
            print(f"   💡 대안: GUI 모드로 시각화하려면 --save_plot 옵션을 제거하세요.")
        return

    # 2) 일반 윈도우 모드 (GUI 가능 환경)
    try:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)  # type: ignore
    except Exception as e:
        print("⚠️ Open3D GUI 창 생성에 실패했습니다. (Headless 환경으로 판단)\n   → 오류 메시지:", e)
        print("대신 오프스크린 모드로 이미지를 저장합니다. '--save_plot <경로>' 인자를 지정하세요.")


def visualize_all_samples_individually(gaussian_boxes: Optional[EvalBoxes] = None, 
                                      pred_boxes: Optional[EvalBoxes] = None, 
                                      gt_boxes: Optional[EvalBoxes] = None, 
                                      scene_name: Optional[str] = None,
                                      score_threshold: Optional[float] = None, 
                                      save_dir: Optional[str] = None, 
                                      max_boxes: int = -1,
                                      max_samples: int = -1,
                                      show_lidar: bool = False,
                                      nusc: Optional[NuScenes] = None,
                                      max_lidar_points: int = 50000) -> None:
    """모든 sample을 개별적으로 시각화합니다.
    
    Args:
        gaussian_boxes: Gaussian 박스들
        pred_boxes: 예측 박스들  
        gt_boxes: Ground truth 박스들
        scene_name: scene 이름
        score_threshold: score threshold
        save_dir: 저장할 디렉토리 (지정하면 각 sample별로 파일 저장)
        max_boxes: 샘플당 최대 박스 개수
        max_samples: 처리할 최대 샘플 개수
        show_lidar: LiDAR 포인트 클라우드 표시 여부
        nusc: NuScenes 객체 (LiDAR 데이터 로딩에 필요)
        max_lidar_points: 최대 LiDAR 포인트 개수
    """
    sample_tokens = []

    # 시간 순서대로 sample_tokens 수집
    if nusc and scene_name:
        # scene 정보에서 시간 순서대로 sample_tokens 가져오기
        scene_token = None
        for scene in nusc.scene:
            if scene['name'] == scene_name:
                scene_token = scene['token']
                break
        
        if scene_token:
            # 해당 scene 찾기
            scene = nusc.get('scene', scene_token)
            
            # scene의 첫 번째 샘플부터 시작하여 시간순으로 수집
            sample = nusc.get('sample', scene['first_sample_token'])
            scene_sample_tokens = []
            
            while True:
                scene_sample_tokens.append(sample['token'])
                if sample['next'] == '':
                    break
                sample = nusc.get('sample', sample['next'])
            
            # 수집된 scene의 sample_tokens 중에서 실제 박스 데이터가 있는 것만 필터링
            available_sample_tokens = set()
            for boxes in [gaussian_boxes, pred_boxes, gt_boxes]:
                if boxes is not None:
                    available_sample_tokens.update(boxes.sample_tokens)
            
            # 시간순으로 정렬된 sample_tokens 중에서 실제 데이터가 있는 것만 유지
            sample_tokens = [token for token in scene_sample_tokens if token in available_sample_tokens]
        else:
            print(f"⚠️ Scene '{scene_name}'을 찾을 수 없습니다.")
            return
    else:
        # nusc나 scene_name이 없는 경우 기존 방식 사용 (순서 보장 안됨)
        if gt_boxes:    
            sample_tokens = list(gt_boxes.sample_tokens)
        elif pred_boxes:
            sample_tokens = list(pred_boxes.sample_tokens)
        elif gaussian_boxes:
            sample_tokens = list(gaussian_boxes.sample_tokens)
    
    if max_samples > 0:
        sample_tokens = sample_tokens[:max_samples]
    
    print(f"📊 총 {len(sample_tokens)}개의 sample을 개별적으로 시각화합니다...")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 결과 파일들을 '{save_dir}' 디렉토리에 저장합니다.")
    
    for i, sample_token in enumerate(sample_tokens):
        print(f"🎯 Processing sample {i+1}/{len(sample_tokens)}: {sample_token}")
        
        save_path = None
        if save_dir:
            # scene 이름으로 하위 폴더 생성
            if scene_name:
                scene_dir = os.path.join(save_dir, scene_name)
                os.makedirs(scene_dir, exist_ok=True)
                # 2자리 인덱스로 파일명 생성
                save_path = os.path.join(scene_dir, f"sample_{i:02d}_{sample_token}.png")
            else:
                # scene_name이 없는 경우 기본 save_dir 사용
                save_path = os.path.join(save_dir, f"sample_{i:02d}_{sample_token}.png")
        
        visualize_ego_translations_open3d(
            gaussian_boxes=gaussian_boxes,
            pred_boxes=pred_boxes, 
            gt_boxes=gt_boxes,
            scene_name=scene_name,
            score_threshold=score_threshold,
            save_path=save_path,
            max_boxes=max_boxes,
            sample_token=sample_token,
            show_lidar=show_lidar,
            nusc=nusc,
            max_lidar_points=max_lidar_points
        )
        
        if not save_dir:
            # GUI 모드일 때는 사용자 입력 대기
            input("다음 sample로 이동하려면 Enter를 누르세요... (Ctrl+C로 종료)")

# -----------------------------------------------------------------------------
# 박스 필터링 유틸리티 (Score / Scene)
# -----------------------------------------------------------------------------

def filter_boxes_by_score(boxes: EvalBoxes, score_threshold: float) -> EvalBoxes:
    """detection_score가 threshold 이상인 boxes만 필터링합니다.

    Args:
        boxes: 필터링할 EvalBoxes
        score_threshold: score threshold (이 값 이상인 박스들만 유지)

    Returns:
        필터링된 EvalBoxes
    """
    filtered_boxes = EvalBoxes()
    total_boxes = 0
    filtered_count = 0

    for sample_token in boxes.sample_tokens:
        sample_boxes = []
        for box in boxes[sample_token]:
            total_boxes += 1
            if hasattr(box, 'detection_score') and box.detection_score >= score_threshold:
                sample_boxes.append(box)
                filtered_count += 1

        if sample_boxes:  # 필터링된 박스가 있는 경우만 추가
            filtered_boxes.add_boxes(sample_token, sample_boxes)

    print(f"✅ Score {score_threshold} 이상인 박스: {filtered_count}/{total_boxes}개")
    return filtered_boxes


def filter_boxes_by_scene(nusc: NuScenes, boxes: EvalBoxes, scene_name: str) -> EvalBoxes:
    """특정 scene에 해당하는 boxes만 필터링합니다.

    Args:
        nusc: NuScenes 객체
        boxes: 필터링할 EvalBoxes
        scene_name: 필터링할 scene 이름 (예: 'scene-0061')

    Returns:
        필터링된 EvalBoxes
    """
    # scene 이름으로 scene 찾기
    scene_token = None
    for scene in nusc.scene:
        if scene['name'] == scene_name:
            scene_token = scene['token']
            break

    if scene_token is None:
        print(f"⚠️ Scene '{scene_name}'을 찾을 수 없습니다.")
        return EvalBoxes()

    # 해당 scene의 sample_tokens 가져오기
    scene_sample_tokens = []
    for sample_token in boxes.sample_tokens:
        sample = nusc.get('sample', sample_token)
        if sample['scene_token'] == scene_token:
            scene_sample_tokens.append(sample_token)

    # 필터링된 박스들로 새로운 EvalBoxes 생성
    filtered_boxes = EvalBoxes()
    for sample_token in scene_sample_tokens:
        if sample_token in boxes.sample_tokens:
            filtered_boxes.add_boxes(sample_token, boxes[sample_token])

    print(f"✅ Scene '{scene_name}'에서 {len(scene_sample_tokens)}개의 샘플을 찾았습니다.")
    return filtered_boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian, Prediction, and Ground Truth boxes")
    parser.add_argument(
        "--gaussian_boxes",
        type=str,
        default='/workspace/drivestudio/output/feasibility_check/updated/poses_selected_tar_selected_src.json',
        help="Path to gaussian boxes json file",
    )
    parser.add_argument(
        "--pred_boxes",
        type=str,
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_selected_tar_selected_tar.json',
        help="Path to prediction boxes json file",
    )
    parser.add_argument(
        "--gt_boxes",
        type=str,
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_gt_pred_selected_src_selected_tar.json',
        help="Path to ground truth boxes json file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="NuScenes version",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/raw",
        help="NuScenes dataroot",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        # default=None,
        default='/workspace/drivestudio/output/feasibility_check/updated/plots',
        help="Path to save the 3D visualization plot"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default='scene-0061',
        help="Scene name to filter boxes"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum detection score threshold for prediction boxes"
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=1500,
        help="시각화할 최대 박스 개수 (<=0 이면 제한 없음)"
    )
    parser.add_argument(
        "--sample_token",
        type=str,
        default=None,
        help="특정 sample만 시각화 (sample token 지정)"
    )
    parser.add_argument(
        "--visualize_individual_samples",
        type=bool,
        default=True,
        help="모든 sample을 개별적으로 시각화"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="개별 시각화할 최대 sample 개수 (<=0 이면 제한 없음)"
    )
    parser.add_argument(
        "--show_lidar",
        type=bool,
        default=True,
        help="LiDAR 포인트 클라우드도 함께 시각화"
    )
    parser.add_argument(
        "--max_lidar_points",
        type=int,
        default=50000,
        help="시각화할 최대 LiDAR 포인트 개수"
    )

    args = parser.parse_args()

    # 최소 하나의 박스 파일은 제공되어야 함
    if not any([args.gaussian_boxes, args.pred_boxes, args.gt_boxes]):
        print("⚠️ 최소 하나의 박스 파일을 제공해야 합니다 (--gaussian_boxes, --pred_boxes, --gt_boxes 중 하나)")
        return

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    
    config = config_factory('detection_cvpr_2019')

    gaussian_boxes = None
    pred_boxes = None
    gt_boxes = None

    # Load gaussian boxes if provided
    if args.gaussian_boxes and os.path.exists(args.gaussian_boxes):
        print(f"📊 Gaussian boxes 로딩 중: {args.gaussian_boxes}")
        gaussian_boxes, _ = load_prediction(args.gaussian_boxes, 
                                           config.max_boxes_per_sample, 
                                           DetectionBox,
                                           verbose=args.verbose)

    # Load prediction boxes if provided
    if args.pred_boxes and os.path.exists(args.pred_boxes):
        print(f"📊 Prediction boxes 로딩 중: {args.pred_boxes}")
        pred_boxes, _ = load_prediction(args.pred_boxes, 
                                       config.max_boxes_per_sample, 
                                       DetectionBox,
                                       verbose=args.verbose)

    # Load ground truth boxes if provided
    if args.gt_boxes and os.path.exists(args.gt_boxes):
        print(f"📊 Ground truth boxes 로딩 중: {args.gt_boxes}")
        gt_boxes, _ = load_prediction(args.gt_boxes, 
                                     config.max_boxes_per_sample, 
                                     DetectionBox,
                                     verbose=args.verbose)

    # Filter by score threshold if prediction boxes exist
    if pred_boxes and args.score_threshold > 0:
        print(f"📊 Score threshold {args.score_threshold}로 pred_boxes 필터링 중...")
        pred_boxes = filter_boxes_by_score(pred_boxes, args.score_threshold)
    
    # Filter boxes by scene
    if args.scene_name:
        if gaussian_boxes:
            gaussian_boxes = filter_boxes_by_scene(nusc, gaussian_boxes, args.scene_name)
        if pred_boxes:
            pred_boxes = filter_boxes_by_scene(nusc, pred_boxes, args.scene_name)
        if gt_boxes:
            gt_boxes = filter_boxes_by_scene(nusc, gt_boxes, args.scene_name)

    # Add ego pose information to all boxes
    if gaussian_boxes:
        gaussian_boxes = add_ego_pose(nusc, gaussian_boxes)
    if pred_boxes:
        pred_boxes = add_ego_pose(nusc, pred_boxes)
    if gt_boxes:
        gt_boxes = add_ego_pose(nusc, gt_boxes)

    # Open3D 3D 시각화
    if args.visualize_individual_samples:
        # 모든 sample을 개별적으로 시각화
        print("모든 sample을 개별적으로 시각화합니다...")
        visualize_all_samples_individually(
            gaussian_boxes=gaussian_boxes,
            pred_boxes=pred_boxes, 
            gt_boxes=gt_boxes,
            scene_name=args.scene_name,
            score_threshold=args.score_threshold,
            save_dir=args.save_plot,
            max_boxes=args.max_boxes,
            max_samples=args.max_samples,
            show_lidar=args.show_lidar,
            nusc=nusc,
            max_lidar_points=args.max_lidar_points
        )
    else:
        # 기존 방식: 모든 박스를 한번에 시각화 또는 특정 sample만 시각화
        print("Open3D 3D 시각화를 생성하고 있습니다...")
        visualize_ego_translations_open3d(
            gaussian_boxes=gaussian_boxes,
            pred_boxes=pred_boxes, 
            gt_boxes=gt_boxes, 
            scene_name=args.scene_name, 
            score_threshold=args.score_threshold, 
            save_path=args.save_plot, 
            max_boxes=args.max_boxes,
            sample_token=args.sample_token,
            show_lidar=args.show_lidar,
            nusc=nusc,
            max_lidar_points=args.max_lidar_points
        )

if __name__ == "__main__":
    main()
    
