import argparse
import json
import os
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import numpy as np
import tqdm
from pyquaternion import Quaternion
import open3d as o3d
from open3d.visualization import rendering
from open3d.visualization.rendering import Camera as O3DCamera  # type: ignore

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

# Constants for Open3D box edges (12 edges for a 3D box)
OPEN3D_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]


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
                box.ego_translation = (float(ego_translation_array[0]), float(ego_translation_array[1]), float(ego_translation_array[2]))
                # Add ego_rotation attribute dynamically
                setattr(box, 'ego_rotation', (ego_rotation.w, ego_rotation.x, ego_rotation.y, ego_rotation.z))
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



# ===========================
# Open3D Visualization Utilities
# ===========================

# -----------------------------
# NEW: Helper for front center sphere
# -----------------------------

def create_open3d_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """Create a sphere mesh with specified center and color.

    Args:
        center (np.ndarray): Center coordinates (3,)
        radius (float): Sphere radius
        color (Tuple[float, float, float]): RGB color (0~1)

    Returns:
        o3d.geometry.TriangleMesh: Sphere mesh for visualization
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    # No need to compute normals for unlit shader
    return sphere

def create_open3d_box(corners: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """Create a 3D box from 8 corner points using thick lines (cylinders).

    Args:
        corners (np.ndarray): (8, 3) corner coordinates
        color (Tuple[float, float, float]): RGB color (0~1)

    Returns:
        o3d.geometry.TriangleMesh: Box mesh with thick edges for visualization
    """
    # Line thickness setting
    line_radius = 0.05
    
    # Mesh to combine all cylinders
    combined_mesh = o3d.geometry.TriangleMesh()
    
    # Create cylinders for 12 edges
    for line_indices in OPEN3D_BOX_LINES:
        start_point = corners[line_indices[0]]
        end_point = corners[line_indices[1]]
        
        # Calculate distance between points
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 1e-6:  # Skip very short lines
            continue
            
        # Create cylinder (initially oriented along Z axis)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=line_radius, 
            height=line_length,
            resolution=8  # Lower resolution for better performance
        )
        
        # Rotate cylinder to correct orientation
        z_axis = np.array([0, 0, 1])
        line_direction = line_vector / line_length
        
        # Calculate rotation axis (cross product)
        rotation_axis = np.cross(z_axis, line_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:  # Not parallel
            # Calculate rotation angle
            cos_angle = np.dot(z_axis, line_direction)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Normalize rotation axis
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Create rotation matrix
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle
            )
            
            # Apply rotation
            cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        
        # Translate cylinder to correct position
        cylinder_center = (start_point + end_point) / 2
        cylinder.translate(cylinder_center)
        
        # Apply color
        cylinder.paint_uniform_color(color)
        
        # Combine meshes
        combined_mesh += cylinder
    
    # No need to compute normals for unlit shader
    return combined_mesh


def load_lidar_pointcloud(nusc: NuScenes, sample_token: str) -> Optional[np.ndarray]:
    """Load LiDAR point cloud from NuScenes sample and transform to ego vehicle coordinates.
    
    Args:
        nusc: NuScenes object
        sample_token: Sample token
        
    Returns:
        Point cloud in ego vehicle coordinates as numpy array (N, 3) or None
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

    # Coordinate frame 추가 (크기 증가)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    # 좌표계는 이미 법선 벡터가 있음
    geometries.append(coord_frame)

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
                                               color=(0.25, 0.25, 0.25),  # 회색
                                               max_points=max_lidar_points)
            geometries.append(lidar_pcd)
        else:
            print("⚠️ LiDAR 포인트 클라우드를 로드할 수 없습니다.")
    elif show_lidar and sample_token is None:
        print("⚠️ LiDAR 시각화는 특정 sample이 지정된 경우에만 가능합니다. --sample_token을 사용하세요.")

    # 특정 sample 시각화 시 ego 좌표계 사용, 전체 시각화 시 global 좌표계 사용
    use_ego_coordinates = sample_token is not None and len(sample_tokens_to_process) == 1

    # 박스들을 geometries에 추가
    gaussian_geometries, gaussian_count = _add_boxes_to_geometries(
        gaussian_boxes, sample_tokens_to_process, center_translation, 
        use_ego_coordinates, (1.0, 0.0, 0.0), max_boxes)  # Red
    geometries.extend(gaussian_geometries)
    
    remaining_boxes = max_boxes - gaussian_count if max_boxes > 0 else -1
    pred_geometries, pred_count = _add_boxes_to_geometries(
        pred_boxes, sample_tokens_to_process, center_translation, 
        use_ego_coordinates, (0.0, 0.0, 1.0), remaining_boxes)  # Blue
    geometries.extend(pred_geometries)
    
    remaining_boxes = max_boxes - gaussian_count - pred_count if max_boxes > 0 else -1
    gt_geometries, gt_count = _add_boxes_to_geometries(
        gt_boxes, sample_tokens_to_process, center_translation, 
        use_ego_coordinates, (0.0, 0.0, 0.0), remaining_boxes)  # Black
    geometries.extend(gt_geometries)

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
    # 시각화 (오프스크린)
    # ---------------------------
    if save_path is not None:
        try:
            # 저장 경로 디렉토리 생성
            save_dir_path = os.path.dirname(save_path)
            if save_dir_path and not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path, exist_ok=True)
                print(f"📁 디렉토리 생성: {save_dir_path}")
            
            print(f"🎨 오프스크린 렌더링 시작... ({len(geometries)}개 객체)")           
            
            # 이미지 렌더링 및 저장
            success = render_and_save_offscreen(geometries, save_path, view_width_m=100)
            
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
            print(f"   💡 대안: GUI 모드로 시각화하려면 --save_dir 옵션을 제거하세요.")
        return

    # 2) 일반 윈도우 모드 (GUI 가능 환경)
    print("⚠️ Headless 환경에서는 GUI 모드를 사용할 수 없습니다.")
    print("💡 이미지를 저장하려면 '--save_dir <경로>' 옵션을 사용하세요.")


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
    # 시간 순서대로 sample_tokens 수집
    if nusc and scene_name:
        scene_sample_tokens = _get_scene_sample_tokens_chronologically(nusc, scene_name)
        if not scene_sample_tokens:
            print(f"⚠️ Scene '{scene_name}'을 찾을 수 없습니다.")
            return
        
        # 실제 박스 데이터가 있는 sample_tokens만 필터링
        available_sample_tokens = _get_available_sample_tokens(gaussian_boxes, pred_boxes, gt_boxes)
        sample_tokens = [token for token in scene_sample_tokens if token in available_sample_tokens]
    else:
        # nusc나 scene_name이 없는 경우 기존 방식 사용 (순서 보장 안됨)
        if gt_boxes:    
            sample_tokens = list(gt_boxes.sample_tokens)
        elif pred_boxes:
            sample_tokens = list(pred_boxes.sample_tokens)
        elif gaussian_boxes:
            sample_tokens = list(gaussian_boxes.sample_tokens)
        else:
            sample_tokens = []
    
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
            save_path = _create_save_path(save_dir, scene_name, i, sample_token)
        
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
            print("⚠️ --save_dir 옵션이 지정되지 않았습니다. Headless 환경에서는 이미지 저장이 필요합니다.")
            break

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


def _get_scene_sample_tokens_chronologically(nusc: NuScenes, scene_name: str) -> List[str]:
    """Scene의 sample_tokens를 시간순으로 가져옵니다.
    
    Args:
        nusc: NuScenes 객체
        scene_name: scene 이름
        
    Returns:
        시간순으로 정렬된 sample_tokens 리스트
    """
    # scene 찾기
    scene_token = None
    for scene in nusc.scene:
        if scene['name'] == scene_name:
            scene_token = scene['token']
            break
    
    if not scene_token:
        return []
    
    # scene의 첫 번째 샘플부터 시작하여 시간순으로 수집
    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    scene_sample_tokens = []
    
    while True:
        scene_sample_tokens.append(sample['token'])
        if sample['next'] == '':
            break
        sample = nusc.get('sample', sample['next'])
    
    return scene_sample_tokens


def _get_available_sample_tokens(gaussian_boxes: Optional[EvalBoxes], 
                                pred_boxes: Optional[EvalBoxes], 
                                gt_boxes: Optional[EvalBoxes]) -> set:
    """사용 가능한 sample_tokens를 모든 박스 타입에서 수집합니다.
    
    Args:
        gaussian_boxes: Gaussian 박스들
        pred_boxes: 예측 박스들
        gt_boxes: Ground truth 박스들
        
    Returns:
        사용 가능한 sample_tokens의 set
    """
    available_sample_tokens = set()
    for boxes in [gaussian_boxes, pred_boxes, gt_boxes]:
        if boxes is not None:
            available_sample_tokens.update(boxes.sample_tokens)
    return available_sample_tokens

def render_and_save_offscreen(geometries, save_path, w=3840, h=2160, view_width_m: Optional[float] = None, view_height_m: Optional[float] = None):
    """Perform offscreen rendering and save image.
    
    Args:
        geometries: Geometric objects to render
        save_path: File path to save
        w: Image width
        h: Image height
        view_width_m: Custom view width in meters
        view_height_m: Custom view height in meters
        
    Returns:
        bool: Success status
    """
    try:
        print(f"Creating offscreen renderer... ({w}x{h})")
        renderer = rendering.OffscreenRenderer(w, h)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 2.0])  # White background

        print("Setting up materials...")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.point_size = 5.0

        print(f"Adding {len(geometries)} objects to scene...")
        valid_objects = 0
        for i, g in enumerate(geometries):
            try:
                scene.add_geometry(f"g{i}", g, mat)
                valid_objects += 1
            except Exception as e:
                print(f"Failed to add object {i}: {e}")
                continue

        print(f"Successfully added {valid_objects}/{len(geometries)} objects.")
        
        if valid_objects == 0:
            print("No objects added. Stopping rendering.")
            return False

        print("Setting up camera...")
        
        # 먼저 모든 객체들의 바운딩 박스를 계산
        all_points = []
        for geom in geometries:
            if hasattr(geom, 'get_axis_aligned_bounding_box'):
                bbox = geom.get_axis_aligned_bounding_box()
                points = np.asarray(bbox.get_box_points())
                all_points.extend(points)
         
        if all_points:
            all_points = np.array(all_points)
            center = np.array([0, 0, 0])
            
            # 카메라 위치 설정 (위에서 아래로 보는 시점)
            camera_pos = center + [0, 0, 50.]
            
            # 카메라 설정 적용
            try:
                scene.camera.look_at(center.tolist(),           # 바라볼 지점
                                camera_pos.tolist(),        # 카메라 위치
                                [0, 1, 0])                 # up 벡터
                print("✅ 카메라 설정 성공")

                # -----------------------------
                # NEW: force orthographic projection (top-down view)
                # -----------------------------
                try:
                    # Calculate a square crop that fully contains the scene (slightly padded)
                    min_bound = np.min(all_points, axis=0)
                    max_bound = np.max(all_points, axis=0)
                    extent = max_bound - min_bound  # [dx, dy, dz]

                    # Renderer aspect ratio (width / height)
                    aspect = float(w) / float(h)

                    # ---- Custom view range support ----
                    if (view_width_m is not None) or (view_height_m is not None):
                        # At least one side specified
                        if (view_width_m is not None) and (view_height_m is not None):
                            half_width  = view_width_m * 0.5
                            half_height = view_height_m * 0.5
                        elif view_width_m is not None:
                            half_width  = view_width_m * 0.5
                            half_height = half_width / aspect
                        else:  # only height specified
                            half_height = view_height_m * 0.5  # type: ignore
                            half_width  = half_height * aspect
                    else:
                        # Automatic range with padding
                        pad = 1.1  # 10 % padding
                        half_width_req  = extent[0] * 0.5 * pad + 0.5
                        half_height_req = extent[1] * 0.5 * pad + 0.5

                        if (half_width_req / half_height_req) >= aspect:
                            half_width  = half_width_req
                            half_height = half_width / aspect
                        else:
                            half_height = half_height_req
                            half_width  = half_height * aspect

                    left,  right = -half_width,  half_width
                    bottom, top  = -half_height, half_height
                    near, far = 0.1, float(extent[2] + 100.0)

                    # Orthographic projection matrix (OpenGL style)               
                    proj = np.array([
                        [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                        [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                        [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                        [0.0, 0.0, 0.0, 1.0],
                    ], dtype=np.float32)

                    # Apply orthographic projection using Open3D API (enum + frustum)
                    scene.camera.set_projection(
                        O3DCamera.Projection.Ortho,
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    )
                    print("✅ Orthographic projection enabled")
                except Exception as e:
                    # Fallback silently if current Open3D version does not support set_projection.
                    print(f"⚠️ Unable to set orthographic projection: {e}")
                # -----------------------------
            except Exception as e:
                print(f"⚠️ 카메라 설정 실패: {e}")
        else:
            print("⚠️ 객체 바운딩 박스를 계산할 수 없어 기본 카메라 설정 사용")

        print("📸 이미지 렌더링 중...")
        img = renderer.render_to_image()
        
        # 렌더링된 이미지 유효성 검사
        if img is None:
            print("❌ 렌더링된 이미지가 None입니다.")
            return False
            
        img_array = np.asarray(img)
        if img_array.size == 0:
            print("❌ 렌더링된 이미지가 비어있습니다.")
            return False
            
        print(f"📊 렌더링된 이미지 크기: {img_array.shape}")
        
        # 이미지 저장
        print(f"💾 이미지 저장 중: {save_path}")
        success = o3d.io.write_image(save_path, img)
        
        if success:
            print(f"✅ 이미지 저장 성공!")
            return True
        else:
            print(f"❌ o3d.io.write_image가 False를 반환했습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 오프스크린 렌더링 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 리소스 정리
        try:
            if 'renderer' in locals():
                del renderer
        except:
            pass

def _add_boxes_to_geometries(boxes: Optional[EvalBoxes], 
                            sample_tokens_to_process: List[str],
                            center_translation: np.ndarray,
                            use_ego_coordinates: bool,
                            color: Tuple[float, float, float],
                            max_boxes: int) -> Tuple[List, int]:
    """박스들을 기하학적 객체로 변환하여 geometries에 추가합니다.
    
    Args:
        boxes: 박스 데이터
        sample_tokens_to_process: 처리할 sample_tokens
        center_translation: 기준 중심점
        use_ego_coordinates: ego 좌표계 사용 여부
        color: 박스 색상
        max_boxes: 최대 박스 개수
        
    Returns:
        (geometries 리스트, 추가된 박스 개수)
    """
    geometries = []
    box_count = 0
    
    if boxes is None:
        return geometries, box_count
    
    for sample_token_iter in sample_tokens_to_process:
        if sample_token_iter not in boxes.sample_tokens:
            continue
            
        for box in boxes[sample_token_iter]:
            if max_boxes > 0 and box_count >= max_boxes:
                break
                
            if not (hasattr(box, 'translation') and box.translation is not None and
                    hasattr(box, 'size') and box.size is not None and
                    hasattr(box, 'rotation') and box.rotation is not None):
                continue
            
            # 좌표계 선택
            if use_ego_coordinates and hasattr(box, 'ego_translation') and hasattr(box, 'ego_rotation'):
                translation = box.ego_translation  # type: ignore
                rotation = getattr(box, 'ego_rotation')  # type: ignore
                relative_translation = np.array(translation)
            else:
                relative_translation = np.array(box.translation) - center_translation
                rotation = box.rotation
            
            # 박스 생성 및 추가
            corners = get_box_corners(relative_translation, box.size, rotation)
            geometries.append(create_open3d_box(corners, color))
            
            # 앞면 중심점 시각화 (크기 증가)
            front_center = (corners[1] + corners[6]) / 2 
            geometries.append(create_open3d_sphere(front_center, radius=0.3, color=color))
            
            box_count += 1
        
        if max_boxes > 0 and box_count >= max_boxes:
            break
    
    return geometries, box_count


def _create_save_path(save_dir: str, scene_name: Optional[str], 
                     sample_index: int, sample_token: str) -> str:
    """저장 경로를 생성합니다.
    
    Args:
        save_dir: 저장 디렉토리
        scene_name: scene 이름 (있으면 하위 폴더 생성)
        sample_index: 샘플 인덱스
        sample_token: 샘플 토큰
        
    Returns:
        생성된 저장 경로
    """
    if scene_name:
        scene_dir = os.path.join(save_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        return os.path.join(scene_dir, f"sample_{sample_index:02d}_{sample_token}.png")
    else:
        return os.path.join(save_dir, f"sample_{sample_index:02d}_{sample_token}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian, Prediction, and Ground Truth boxes")
    parser.add_argument(
        "--gaussian_boxes",
        type=str,
        default='/workspace/drivestudio/output/feasibility_check/updated/poses_selected_tar2.json',
        help="Path to gaussian boxes json file",
    )
    parser.add_argument(
        "--pred_boxes",
        type=str,
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred_selected_tar1.json',
        help="Path to prediction boxes json file",
    )
    parser.add_argument(
        "--gt_boxes",
        type=str,
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_gt_pred_selected_src.json',
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
        "--save_dir",
        type=str,
        # default=None,
        default='/workspace/drivestudio/output/test/plots',
        help="Path to save the 3D visualization plot"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default='scene-0061',
        # default=None,
        help="Scene name to visualize boxes (e.g., 'scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', "
             "'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', 'scene-1094', 'scene-1100')",
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
        default=2000,
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

    # At least one box file must be provided
    if not any([args.gaussian_boxes, args.pred_boxes, args.gt_boxes]):
        print("At least one box file must be provided (--gaussian_boxes, --pred_boxes, or --gt_boxes)")
        return

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    
    config = config_factory('detection_cvpr_2019')

    gaussian_boxes = None
    pred_boxes = None
    gt_boxes = None

    # Load gaussian boxes if provided
    if args.gaussian_boxes and os.path.exists(args.gaussian_boxes):
        print(f"Loading Gaussian boxes: {args.gaussian_boxes}")
        gaussian_boxes, _ = load_prediction(args.gaussian_boxes, 
                                           config.max_boxes_per_sample, 
                                           DetectionBox,
                                           verbose=args.verbose)

    # Load prediction boxes if provided
    if args.pred_boxes and os.path.exists(args.pred_boxes):
        print(f"Loading prediction boxes: {args.pred_boxes}")
        pred_boxes, _ = load_prediction(args.pred_boxes, 
                                       config.max_boxes_per_sample, 
                                       DetectionBox,
                                       verbose=args.verbose)

    # Load ground truth boxes if provided
    if args.gt_boxes and os.path.exists(args.gt_boxes):
        print(f"Loading ground truth boxes: {args.gt_boxes}")
        gt_boxes, _ = load_prediction(args.gt_boxes, 
                                     config.max_boxes_per_sample, 
                                     DetectionBox,
                                     verbose=args.verbose)

    # Filter by score threshold if prediction boxes exist
    if pred_boxes and args.score_threshold > 0:
        print(f"Filtering pred_boxes with score threshold {args.score_threshold}...")
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
        print("Visualizing all samples individually...")
        visualize_all_samples_individually(
            gaussian_boxes=gaussian_boxes,
            pred_boxes=pred_boxes, 
            gt_boxes=gt_boxes,
            scene_name=args.scene_name,
            score_threshold=args.score_threshold,
            save_dir=args.save_dir,
            max_boxes=args.max_boxes,
            max_samples=args.max_samples,
            show_lidar=args.show_lidar,
            nusc=nusc,
            max_lidar_points=args.max_lidar_points
        )
    else:
        # Default: visualize all boxes at once or specific sample
        print("Creating Open3D 3D visualization...")
        visualize_ego_translations_open3d(
            gaussian_boxes=gaussian_boxes,
            pred_boxes=pred_boxes, 
            gt_boxes=gt_boxes, 
            scene_name=args.scene_name, 
            score_threshold=args.score_threshold, 
            save_path=args.save_dir, 
            max_boxes=args.max_boxes,
            sample_token=args.sample_token,
            show_lidar=args.show_lidar,
            nusc=nusc,
            max_lidar_points=args.max_lidar_points
        )

if __name__ == "__main__":
    main()
    
