import argparse
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import json
import os
from pathlib import Path
from pyquaternion import Quaternion
import open3d as o3d
from open3d.visualization import rendering
from open3d.visualization.rendering import Camera as O3DCamera  # type: ignore
import torch
import glob
import re
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation as R

# ===========================
# Open3D 시각화 유틸리티
# ===========================

# Open3D LineSet 생성을 위한 에지 인덱스 (12개)
OPEN3D_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 아래면
    [4, 5], [5, 6], [6, 7], [7, 4],  # 위면
    [0, 4], [1, 5], [2, 6], [3, 7]   # 옆면
]

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
    corners_rotated = (rotation_matrix @ corners_local.T).T
    
    # 평행이동 적용
    corners_world = corners_rotated + np.array(translation)
    
    return corners_world

def create_open3d_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """지정한 중심과 색상의 구(Sphere) Mesh를 생성합니다."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def create_open3d_box(corners: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """8개 꼭짓점 정보로부터 Open3D 두꺼운 선으로 이루어진 육면체 객체를 생성합니다."""
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

def render_and_save_offscreen(geometries, save_path, w=1920, h=1080, view_width_m: Optional[float] = None, view_height_m: Optional[float] = None):
    """오프스크린 렌더링을 수행하고 이미지를 저장합니다.
    
    Args:
        geometries: 렌더링할 기하학적 객체들
        save_path: 저장할 파일 경로
        w: 이미지 폭
        h: 이미지 높이
        view_width_m: 커스텀 뷰 폭 (미터)
        view_height_m: 커스텀 뷰 높이 (미터)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 싱글톤 렌더러 가져오기 (VRAM 누수 방지)
        renderer = _get_offscreen_renderer(w, h)

        # 흰색 배경 적용 (재사용 시 매 프레임 설정 필요)
        try:
            renderer.scene.set_background([1.0, 1.0, 1.0, 2.0])  # RGBA
        except Exception:
            pass

        valid_objects = 0
        for i, g in enumerate(geometries):
            try:
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.base_color = [1.0, 1.0, 1.0, 1.0]
                mat.point_size = 5.0

                renderer.scene.add_geometry(f"g{i}", g, mat)
                valid_objects += 1
            except Exception as e:
                print(f"객체 {i} 추가 실패: {e}")
                continue
        
        if valid_objects == 0:
            print("추가된 객체가 없습니다. 렌더링을 중단합니다.")
            return False
        
        # 모든 객체들의 바운딩 박스를 계산
        all_points = []
        for geom in geometries:
            if hasattr(geom, 'get_axis_aligned_bounding_box'):
                bbox = geom.get_axis_aligned_bounding_box()
                points = np.asarray(bbox.get_box_points())
                all_points.extend(points)
            else:
                print(f"❌ {geom} 객체에서 바운딩 박스 계산 실패")
         
        if all_points:
            all_points = np.array(all_points)
            center = np.array([0, 0, 0])
            
            # 카메라 위치 설정 (위에서 아래로 보는 시점)
            camera_pos = center + [0, 0, 50.]
            
            # 카메라 설정 적용
            try:
                renderer.scene.camera.look_at(center.tolist(),           # 바라볼 지점
                                camera_pos.tolist(),        # 카메라 위치
                                [0, 1, 0])                 # up 벡터

                # 직교 투영(orthographic projection) 강제 적용
                try:
                    # 씬을 완전히 포함하는 정사각형 크롭 계산 (약간의 패딩 포함)
                    min_bound = np.min(all_points, axis=0)
                    max_bound = np.max(all_points, axis=0)
                    extent = max_bound - min_bound  # [dx, dy, dz]

                    # 렌더러 종횡비 (width / height)
                    aspect = float(w) / float(h)

                    # 커스텀 뷰 범위 지원
                    if (view_width_m is not None) or (view_height_m is not None):
                        # 적어도 한 면이 지정됨
                        if (view_width_m is not None) and (view_height_m is not None):
                            half_width  = view_width_m * 0.5
                            half_height = view_height_m * 0.5
                        elif view_width_m is not None:
                            half_width  = view_width_m * 0.5
                            half_height = half_width / aspect
                        else:  # 높이만 지정됨
                            half_height = view_height_m * 0.5  # type: ignore
                            half_width  = half_height * aspect
                    else:
                        # 패딩을 포함한 자동 범위
                        pad = 1.1  # 10% 패딩
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

                    # Open3D API를 사용한 직교 투영 (enum + frustum)
                    renderer.scene.camera.set_projection(
                        O3DCamera.Projection.Ortho,
                        left,
                        right,
                        bottom,
                        top,
                        near,
                        far,
                    )
                except Exception as e:
                    # 현재 Open3D 버전에서 set_projection을 지원하지 않는 경우 조용히 폴백
                    print(f"⚠️ 직교 투영 설정 불가: {e}")
            except Exception as e:
                print(f"⚠️ 카메라 설정 실패: {e}")
        else:
            print("⚠️ 객체 바운딩 박스를 계산할 수 없어 기본 카메라 설정 사용")

        img = renderer.render_to_image()
        
        # 렌더링된 이미지 유효성 검사
        if img is None:
            print("❌ 렌더링된 이미지가 None입니다.")
            return False
            
        img_array = np.asarray(img)
        if img_array.size == 0:
            print("❌ 렌더링된 이미지가 비어있습니다.")
            return False
                  
        # 이미지 저장
        success = o3d.io.write_image(save_path, img)
        
        if success:
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
        # 다음 프레임을 위해 지오메트리만 정리 (렌더러는 재사용)
        try:
            if 'renderer' in locals():
                renderer.scene.clear_geometry()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# OffscreenRenderer 싱글톤 관리 (반복 생성으로 인한 VRAM 누수 방지)
# ---------------------------------------------------------------------------

_GLOBAL_RENDERER: Optional[rendering.OffscreenRenderer] = None  # 재사용할 렌더러
_GLOBAL_RENDERER_SIZE: Optional[Tuple[int, int]] = None  # (w, h)

def _get_offscreen_renderer(w: int, h: int) -> rendering.OffscreenRenderer:
    """필요 시 새로운 OffscreenRenderer 를 생성하고, 그렇지 않으면 기존 인스턴스를 재사용합니다.

    Open3D <0.18 버전에서는 OffscreenRenderer 를 반복 생성할 때 GPU 메모리가
    해제되지 않는 이슈가 있어, 싱글톤으로 관리하여 누수를 방지합니다.
    """
    global _GLOBAL_RENDERER, _GLOBAL_RENDERER_SIZE

    if _GLOBAL_RENDERER is None or _GLOBAL_RENDERER_SIZE != (w, h):
        # 기존 렌더러를 해제하고 새 인스턴스를 생성
        try:
            if _GLOBAL_RENDERER is not None:
                _GLOBAL_RENDERER.release_resources()  # type: ignore[attr-defined]
        except Exception:
            pass

        _GLOBAL_RENDERER = rendering.OffscreenRenderer(w, h)
        _GLOBAL_RENDERER_SIZE = (w, h)

    # 매 프레임마다 지오메트리 초기화
    try:
        _GLOBAL_RENDERER.scene.clear_geometry()
    except Exception:
        pass

    return _GLOBAL_RENDERER

# ===========================
# 좌표 변환 유틸리티
# ===========================

def get_camera_front_start_pose(nuscenes_dataroot: str, scene_name: str, version: str = 'v1.0-mini') -> Optional[np.ndarray]:
    """NuScenes 첫 번째 카메라 포즈를 가져옴 (좌표 정렬용)"""
    try:
        # NuScenes API 초기화
        nusc = NuScenes(version=version, dataroot=nuscenes_dataroot, verbose=False)
        
        # scene 이름으로 scene 찾기
        scene = None
        for s in nusc.scene:
            if s['name'] == scene_name:
                scene = s
                break
                
        if scene is None:
            print(f"⚠️ Scene '{scene_name}'을 찾을 수 없습니다.")
            return None
            
        # 첫 번째 sample 가져오기
        first_sample_token = scene['first_sample_token']
        first_sample_record = nusc.get('sample', first_sample_token)
        
        # CAM_FRONT (cam_idx=0) 데이터 가져오기
        cam_name = "CAM_FRONT"
        cam_data = nusc.get('sample_data', first_sample_record['data'][cam_name])
        calib_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        # Extrinsics (camera to ego)
        extrinsics_cam_to_ego = np.eye(4)
        extrinsics_cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
        extrinsics_cam_to_ego[:3, 3] = np.array(calib_data['translation'])
        
        # Get ego pose (ego to world)
        ego_pose_data = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
        
        # Transform camera extrinsics to world coordinates
        camera_front_start = ego_to_world @ extrinsics_cam_to_ego
        
        return camera_front_start
        
    except Exception as e:
        print(f"⚠️ NuScenes API를 통한 카메라 포즈 로드 실패: {e}")
        return None

def transform_pose_to_world(translation: np.ndarray, rotation: np.ndarray, camera_front_start: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """obj_to_camera 포즈를 obj_to_world 포즈로 변환"""
    # quaternion을 rotation matrix로 변환
    if len(rotation) == 4:  # quaternion [w, x, y, z]
        rot_matrix = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()  # scipy는 [x,y,z,w] 순서
    else:
        raise ValueError(f"Unsupported rotation format: {rotation}")
    
    # 4x4 변환 행렬 구성 (obj_to_camera)
    obj_to_camera = np.eye(4)
    obj_to_camera[:3, :3] = rot_matrix
    obj_to_camera[:3, 3] = translation
    
    # obj_to_world = camera_front_start @ obj_to_camera
    obj_to_world = camera_front_start @ obj_to_camera
    
    # world 좌표계에서 translation과 rotation 추출
    world_translation = obj_to_world[:3, 3]
    world_rotation_matrix = obj_to_world[:3, :3]
    
    world_rotation_quat = R.from_matrix(world_rotation_matrix).as_quat()  # [x, y, z, w]
    world_rotation_quat = np.array([world_rotation_quat[3], world_rotation_quat[0], world_rotation_quat[1], world_rotation_quat[2]])  # [w, x, y, z]
    
    return world_translation, world_rotation_quat

# ===========================
# NuScenes 박스 및 LiDAR 처리 유틸리티
# ===========================

def load_lidar_pointcloud(nusc: 'NuScenes', sample_token: str) -> Optional[np.ndarray]:
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

def add_ego_pose(nusc: 'NuScenes', eval_boxes: 'EvalBoxes') -> 'EvalBoxes':
    """각 박스에 ego pose 정보를 추가합니다."""
        
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
            ego_translation_array = ego_rotation_global.inverse.rotate(relative_translation_global)
            
            # Step 3: Transform box rotation to ego coordinates
            box_rotation_global = Quaternion(list(box.rotation))  # type: ignore
            ego_rotation = ego_rotation_global.inverse * box_rotation_global
            
            if hasattr(box, 'ego_translation'):
                if isinstance(ego_translation_array, np.ndarray):
                    box.ego_translation = tuple(ego_translation_array.tolist())
                else:
                    box.ego_translation = tuple(ego_translation_array)
                # Add ego_rotation attribute dynamically
                setattr(box, 'ego_rotation', tuple([ego_rotation.w, ego_rotation.x, ego_rotation.y, ego_rotation.z]))  # type: ignore

    return eval_boxes

def add_ego_pose_to_boxes(nusc: 'NuScenes', 
                        frame_boxes: Dict[int, List], 
                        sample_tokens: List[str]) -> Dict[int, List]:
    """체크포인트 박스 데이터에 ego pose 정보를 추가합니다.
    
    Args:
        nusc: NuScenes 객체
        frame_boxes: 체크포인트에서 추출한 프레임별 박스 데이터
        sample_tokens: NuScenes sample 토큰들 (시간순)
        
    Returns:
        ego pose가 추가된 프레임별 박스 데이터
    """
    if not sample_tokens:
        print("⚠️ sample_tokens가 제공되지 않았습니다. ego pose 변환을 건너뜁니다.")
        return frame_boxes
    
    # 5의 배수 프레임만 처리 (NuScenes 키프레임)
    keyframe_indices = [i for i in range(0, len(sample_tokens), 1)]  # sample_tokens는 이미 5의 배수로 제공됨
    
    for frame_id, boxes_data in frame_boxes.items():
        # frame_id를 sample_token 인덱스로 변환 (5의 배수)
        sample_idx = frame_id // 5
        if sample_idx >= len(sample_tokens):
            continue
            
        sample_token = sample_tokens[sample_idx]
        
        try:
            # NuScenes에서 ego pose 정보 가져오기
            sample_rec = nusc.get('sample', sample_token)
            sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

            # Get ego pose transformation
            ego_translation_global = np.array(pose_record['translation'])
            ego_rotation_global = Quaternion(pose_record['rotation'])
            
            # 각 박스에 ego pose 변환 적용
            for box_data in boxes_data:
                # 체크포인트 박스의 translation과 rotation (이미 global 좌표계로 가정)
                box_translation_global = np.array(box_data['translation'])
                box_rotation_global = Quaternion(box_data['rotation'])  # [w, x, y, z]
                
                # Step 1: Get relative position vector in global coordinates
                relative_translation_global = box_translation_global - ego_translation_global
                
                # Step 2: Rotate this position vector to ego vehicle's local coordinate system
                ego_translation_array = ego_rotation_global.inverse.rotate(relative_translation_global)
                
                # Step 3: Transform box rotation to ego coordinates
                ego_rotation = ego_rotation_global.inverse * box_rotation_global
                
                # ego 좌표계 정보를 박스 데이터에 추가
                box_data['ego_translation'] = ego_translation_array.tolist()
                box_data['ego_rotation'] = [ego_rotation.w, ego_rotation.x, ego_rotation.y, ego_rotation.z]
                
        except Exception as e:
            print(f"⚠️ Frame {frame_id}의 ego pose 변환 실패: {e}")
            continue
    
    return frame_boxes

def _add_boxes_to_geometries_from_dict(frame_boxes: Dict[int, List], 
                            frame_id: int,
                            color_mapping: Dict[str, Tuple[float, float, float]],
                            use_ego_coordinates: bool = True) -> List:
    """박스들을 기하학적 객체로 변환합니다."""
    geometries = []
    
    if frame_id not in frame_boxes:
        return geometries
    
    for box_data in frame_boxes[frame_id]:
        # 좌표계 선택
        if use_ego_coordinates and 'ego_translation' in box_data and 'ego_rotation' in box_data:
            translation = box_data['ego_translation']
            rotation = box_data['ego_rotation']
        else:
            translation = box_data['translation']
            rotation = box_data['rotation']
        
        size = box_data['size']
        node_type = box_data.get('node_type', 'Unknown')
        
        # 노드 타입별 색상 선택
        color = color_mapping.get(node_type, (0.5, 0.5, 0.5))  # 기본 회색
        
        # 박스 생성 및 추가
        corners = get_box_corners(translation, size, rotation)
        geometries.append(create_open3d_box(corners, color))
        
        # 앞면 중심점 시각화
        front_center = (corners[1] + corners[6]) / 2 
        geometries.append(create_open3d_sphere(front_center, radius=0.3, color=color))
    
    return geometries

def _add_boxes_to_geometries_from_evalboxes(boxes: Optional['EvalBoxes'], 
                            sample_token: str,
                            color: Tuple[float, float, float],
                            use_ego_coordinates: bool = True) -> List:
    """박스들을 기하학적 객체로 변환합니다."""
    geometries = []
    
    if not boxes or sample_token not in boxes.sample_tokens:
        return geometries
    
    for box in boxes[sample_token]:
        if not (hasattr(box, 'translation') and box.translation is not None and
                hasattr(box, 'size') and box.size is not None and
                hasattr(box, 'rotation') and box.rotation is not None):
            continue
        
        # 좌표계 선택
        if use_ego_coordinates and hasattr(box, 'ego_translation') and hasattr(box, 'ego_rotation'):
            translation = box.ego_translation  # type: ignore
            rotation = getattr(box, 'ego_rotation')  # type: ignore
        else:
            translation = box.translation
            rotation = box.rotation
        
        # 박스 생성 및 추가
        corners = get_box_corners(translation, box.size, rotation)
        geometries.append(create_open3d_box(corners, color))
        
        # 앞면 중심점 시각화
        front_center = (corners[1] + corners[6]) / 2 
        geometries.append(create_open3d_sphere(front_center, radius=0.3, color=color))
    
    return geometries

# ===========================
# 박스 데이터 로딩 및 애니메이션 함수들
# ===========================

def _get_scene_sample_tokens_chronologically(nusc: 'NuScenes', scene_name: str) -> List[str]:
    """Scene의 sample_tokens를 시간순으로 가져옵니다."""
        
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

def create_all_sample_animations(box_poses_dir: str, output_dir: str,
                                scene_name: Optional[str] = None,
                                sample_token: Optional[str] = None,
                                pred_boxes: Optional['EvalBoxes'] = None,
                                gt_boxes: Optional['EvalBoxes'] = None,
                                show_lidar: bool = False,
                                nusc: Optional['NuScenes'] = None,
                                max_lidar_points: int = 50000) -> None:
    """체크포인트들에서 모든 샘플의 박스 최적화 과정을 애니메이션으로 생성합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # box pose JSON 파일 탐색
    box_poses_files = find_boxpose_files(box_poses_dir)

    if not box_poses_files:
        print(f"❌ box pose JSON 파일을 찾을 수 없습니다: {box_poses_dir}")
        return
    
    print(f"🎬 애니메이션 생성 시작: {len(box_poses_files)}개 box pose 파일")
    if show_lidar:
        print(f"📡 LiDAR 포인트 클라우드 포함 (최대 {max_lidar_points:,}개 포인트)")
    
    # 샘플별 애니메이션 생성
    if scene_name or sample_token:
        # ---------- 새로운 구현: 체크포인트를 한 번만 로드해 모든 샘플 처리 ----------

        # 0) 처리할 sample token 목록 결정
        scene_sample_tokens: List[str] = []
        if scene_name and nusc is not None:
            print(f"🎬 Scene '{scene_name}'의 샘플들을 시간순으로 처리합니다")
            scene_sample_tokens = _get_scene_sample_tokens_chronologically(nusc, scene_name)
            if not scene_sample_tokens:
                print(f"❌ Scene '{scene_name}'을 찾을 수 없거나 샘플이 없습니다")
                return
            print(f"🎯 Scene '{scene_name}'에서 {len(scene_sample_tokens)}개 샘플 프레임 발견")
        elif sample_token:
            scene_sample_tokens = [sample_token]
            print(f"🎯 특정 샘플만 처리: {sample_token}")
        else:
            print("⚠️ scene_name과 sample_token이 모두 제공되지 않아 샘플을 결정할 수 없습니다")
            return

        # 1) 샘플 컨텍스트 초기화 (LiDAR, 출력 디렉토리 등)
        sample_contexts: Dict[str, Dict[str, Any]] = {}
        for sample_idx, current_sample_token in enumerate(scene_sample_tokens):
            lidar_points = None
            if show_lidar and nusc is not None:
                lidar_points = load_lidar_pointcloud(nusc, current_sample_token)
                if lidar_points is None:
                    print(f"❌ LiDAR 포인트 로드 실패: {current_sample_token}")

            sample_output_dir = os.path.join(output_dir, f"sample_{sample_idx:02d}_{current_sample_token}")
            os.makedirs(sample_output_dir, exist_ok=True)

            sample_contexts[current_sample_token] = {
                'idx': sample_idx,
                'lidar_points': lidar_points,
                'output_dir': sample_output_dir,
                'frame_images': []
            }
        print(f"\n🎬 샘플 {len(scene_sample_tokens)}개 초기화 완료")

        # 2) camera_front_start (scene 기준) 한 번만 계산
        camera_front_start = None
        if nusc is not None and scene_name:
            camera_front_start = get_camera_front_start_pose(nusc.dataroot, scene_name, nusc.version)
            if camera_front_start is not None:
                print("✅ Camera front start pose 로드 완료 (1회)")

        # 3) 노드 타입별 색상 매핑 (고정)
        color_mapping = {
            'RigidNodes': (1.0, 0.0, 0.0),
            'SMPLNodes': (1.0, 0.2, 0.0),
            'DeformableNodes': (1.0, 0.25, 0.3),
            'Unknown': (1.0, 0.2, 0.11)
        }

        total_iterations = box_poses_files[-1][0]

        # 4) 체크포인트를 순차적으로 처리하며 모든 샘플 이미지 저장
        for bp_idx, (iteration, json_path) in enumerate(box_poses_files):
            print(f"\n🔄 Box poses {bp_idx+1}/{len(box_poses_files)} 처리 중 - Iteration {iteration:06d}")       

            all_frame_boxes = extract_all_boxes_from_json(json_path)
            if all_frame_boxes and nusc is not None and scene_sample_tokens:
                all_frame_boxes = add_ego_pose_to_boxes(nusc, all_frame_boxes, scene_sample_tokens)

            for idx, current_sample_token in enumerate(scene_sample_tokens):
                print(f"{bp_idx+1} / {len(box_poses_files)} 번째 box pose 파일, {idx+1} / {len(scene_sample_tokens)} 번째 샘플 인덱스 처리중...")
                ctx = sample_contexts[current_sample_token]
                sample_idx = ctx['idx']
                sample_output_dir = ctx['output_dir']
                lidar_points = ctx['lidar_points']

                frame_path = os.path.join(sample_output_dir, f"frame_{bp_idx:02d}_iter_{iteration:06d}.png")
                if os.path.exists(frame_path):
                    print(f"skip frame: {frame_path}")
                    ctx['frame_images'].append(frame_path)
                    continue

                # --------- 시각화용 Geometry 생성 ---------
                geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)]

                if lidar_points is not None:
                    lidar_pcd = create_open3d_pointcloud(lidar_points, color=(0.25, 0.25, 0.25), max_points=max_lidar_points)
                    geometries.append(lidar_pcd)

                if all_frame_boxes:
                    frame_id = sample_idx * 5  # 5의 배수로 매핑
                    box_geometries = _add_boxes_to_geometries_from_dict(
                        all_frame_boxes,
                        frame_id,
                        color_mapping,
                        use_ego_coordinates=True
                    )
                    geometries.extend(box_geometries)

                # pred / gt boxes
                if pred_boxes and current_sample_token in pred_boxes.sample_tokens:
                    geometries.extend(_add_boxes_to_geometries_from_evalboxes(pred_boxes, current_sample_token, (0.0, 0.0, 1.0)))
                if gt_boxes and current_sample_token in gt_boxes.sample_tokens:
                    geometries.extend(_add_boxes_to_geometries_from_evalboxes(gt_boxes, current_sample_token, (0.0, 0.0, 0.0)))

                # 렌더링 및 저장
                success = render_and_save_offscreen(geometries, frame_path, w=1920, h=1080, view_width_m=100)
                if success and os.path.exists(frame_path) and os.path.getsize(frame_path) > 1000:
                    if add_text_overlay_to_image(frame_path, scene_name if scene_name else "Unknown Scene", sample_idx, iteration, total_iterations):
                        ctx['frame_images'].append(frame_path)
                    else:
                        ctx['frame_images'].append(frame_path)
                else:
                    print(f"  ❌ 이미지 저장 실패: {frame_path}")

        # 5) GIF 생성 (체크포인트 처리 후)
        for current_sample_token, ctx in sample_contexts.items():
            if ctx['frame_images']:
                animation_name = f"box_optimization_sample_{ctx['idx']:02d}_{current_sample_token}.gif"
                create_gif_animation_from_files(ctx['frame_images'], output_dir, animation_name)
                print(f"  ✅ 샘플 {current_sample_token} 애니메이션 완료: {ctx['output_dir']}")
            else:
                print(f"  ❌ 샘플 {current_sample_token}에 대한 유효한 이미지가 없습니다")

        print(f"\n🎉 모든 애니메이션 생성 완료! 결과: {output_dir}")
        return  # 기존 로직 실행 방지

    print(f"\n🎉 모든 애니메이션 생성 완료! 결과: {output_dir}")

def add_text_overlay_to_image(image_path: str, scene_name: str, frame_idx: int, 
                             iteration: int, total_iterations: int) -> bool:
    """이미지 좌측 상단에 정보 텍스트를 추가합니다."""
    try:
        # 이미지 로드
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font_size = 50
        
        # 폰트 설정 (기본 폰트 사용, 크기 조정)
        try:
            # 시스템에서 사용 가능한 폰트 시도
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                # 대안 폰트 시도
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    # 다른 시스템 폰트들 시도
                    system_fonts = [
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                        "/System/Library/Fonts/Arial.ttf",  # macOS
                        "/Windows/Fonts/arial.ttf",  # Windows
                        "/usr/share/fonts/TTF/arial.ttf"  # Some Linux
                    ]
                    font = None
                    for font_path in system_fonts:
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            break
                        except:
                            continue
                    
                    if font is None:
                        raise Exception("No system fonts found")
                        
                except:
                    # 최후의 수단: 기본 폰트 사용 (크기 조정 불가)
                    font = ImageFont.load_default()
                    print(f"⚠️ 시스템 폰트를 찾을 수 없어 기본 폰트를 사용합니다 (크기 조정 불가)")
        
        # 텍스트 내용 구성 (가로 배치용)
        text_parts = [
            f"Scene: {scene_name}",
            f"Frame: {frame_idx}",
            f"Iteration: {iteration:,} / {total_iterations:,}"
        ]
        
        # 텍스트를 가로로 배치하기 위해 구분자로 연결
        full_text = " | ".join(text_parts)
        
        # 텍스트 위치 설정 (좌측 상단)
        x_offset = 20
        y_offset = 20
        
        # 전체 텍스트의 크기 측정
        bbox = draw.textbbox((0, 0), full_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 반투명 배경 박스 (검은색, 70% 투명도)
        background_box = [
            x_offset - 10, 
            y_offset - 10, 
            x_offset + text_width + 20, 
            y_offset + text_height + 20
        ]
        
        # 배경 박스를 위한 별도 이미지 생성 후 알파 블렌딩
        overlay = Image.new('RGBA', img.size)
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(background_box, fill=(0, 0, 0, 180))  # 검은색, 70% 불투명
        
        # 원본 이미지와 오버레이 합성
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        
        # 흰색 텍스트 그리기 (가로 한 줄)
        draw.text((x_offset, y_offset), full_text, fill=(255, 255, 255, 255), font=font)
        
        # RGB 모드로 변환 후 저장
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img.save(image_path)
        return True
        
    except Exception as e:
        print(f"⚠️ 텍스트 오버레이 추가 실패: {e}")
        return False

def create_gif_animation_from_files(frame_files: List[str], output_dir: str, base_name: str) -> None:
    """프레임 이미지 파일들을 GIF 애니메이션으로 결합합니다."""
    try:
        if not frame_files:
            print(f"❌ 프레임 이미지 파일이 없습니다")
            return
        
        print(f"🔗 {len(frame_files)}개 프레임을 GIF로 결합 중...")
        
        # 이미지들 로드
        images = []
        for frame_file in frame_files:
            if os.path.exists(frame_file) and os.path.getsize(frame_file) > 1000:  # 최소 1KB 이상
                img = Image.open(frame_file)
                images.append(img)
            else:
                print(f"⚠️ 유효하지 않은 프레임 건너뛰기: {frame_file}")
        
        if not images:
            print(f"❌ 유효한 프레임 이미지가 없습니다")
            return
        
        # GIF 경로 생성
        gif_path = os.path.join(output_dir, base_name)
        
        # GIF 생성
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=125,
            loop=0
        )
        
        print(f"✅ GIF 애니메이션 생성 완료: {gif_path}")
        print(f"   📊 프레임 수: {len(images)}개")
        print(f"   📂 크기: {os.path.getsize(gif_path):,} bytes")
        
    except Exception as e:
        print(f"❌ GIF 생성 실패: {e}")

def find_boxpose_files(boxpose_dir: str) -> List[Tuple[int, str]]:
    """box_poses 디렉토리에서 box_poses_*.json 파일을 iteration 순서대로 정렬해서 찾습니다."""

    boxpose_files: List[Tuple[int, str]] = []

    pattern = os.path.join(boxpose_dir, "box_poses_*.json")
    files = glob.glob(pattern)

    for file_path in files:
        filename = os.path.basename(file_path)
        # box_poses_01000.json → 1000 추출
        match = re.search(r'box_poses_(\d+)\.json', filename)
        if match:
            iteration = int(match.group(1))
            boxpose_files.append((iteration, file_path))

    # iteration 순으로 정렬
    boxpose_files.sort(key=lambda x: x[0])

    print(f"📁 찾은 box pose 파일들 ({len(boxpose_files)}개):")
    for iteration, file_path in boxpose_files:
        print(f"  - Iteration {iteration:06d}: {os.path.basename(file_path)}")

    return boxpose_files

def extract_all_boxes_from_json(json_path: str) -> Optional[Dict[int, List]]:
    """box_poses_*.json 파일에서 모든 프레임의 박스 정보를 추출합니다."""

    print(f"🔍 JSON 로딩: {os.path.basename(json_path)}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'results' not in data:
        print(f"❌ 유효하지 않은 JSON 구조: {json_path}")
        return None

    results = data['results']

    frame_boxes: Dict[int, List] = {}

    total_boxes = 0
    for frame_id_str, boxes in results.items():
        try:
            frame_id = int(frame_id_str)
        except ValueError:
            print(f"❌ frame id가 정수형이 아닙니다: {frame_id_str}")
            continue

        # box 리스트를 그대로 저장 (필요 시 deep copy)
        frame_boxes[frame_id] = boxes
        total_boxes += len(boxes)

    print(f"✅ JSON에서 추출된 총 박스 수: {total_boxes}개 ({len(frame_boxes)}개 프레임)")

    return frame_boxes

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize bounding box optimization steps as animation")
    
    # 애니메이션 관련 인자들
    parser.add_argument(
        "--box_poses_dir",
        type=str,
        default="/workspace/drivestudio/output/test_250703/test_try1/box_poses",
        help="Directory containing box pose JSON files (box_poses_*.json)"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default='scene-0103',
        help="Scene name to animate boxes optimization (e.g., 'scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', "
                                                "'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', "
                                                "'scene-1094', 'scene-1100')",
    )
    parser.add_argument(
        "--sample_token",
        type=str,
        default=None,
        help="Specific sample token to visualize (if provided, only this sample will be processed)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save animation frames and final GIF"
    )
    
    # pred_boxes, gt_boxes 관련 인자들
    parser.add_argument(
        "--pred_boxes",
        type=str,
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred_real_selected_tar1.json',
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
    
    # LiDAR 관련 인자들
    parser.add_argument(
        "--show_lidar",
        type=bool,
        default=True,
        help="LiDAR 포인트 클라우드도 함께 시각화"
    )
    parser.add_argument(
        "--max_lidar_points",
        type=int,
        default=500000,
        help="시각화할 최대 LiDAR 포인트 개수"
    )
    
    args = parser.parse_args()
    
    print("🚀 바운딩 박스 최적화 시각화 도구")
    print(f"📁 Box poses 디렉토리: {args.box_poses_dir}")
    print(f"🎬 Scene: {args.scene_name}")
    print(f"🎯 샘플 토큰: {args.sample_token if args.sample_token else '모든 샘플'}")
    print(f"📡 LiDAR 시각화: {'활성화' if args.show_lidar else '비활성화'}")
    
    # NuScenes 초기화 (scene_name, pred_boxes, gt_boxes, LiDAR 기능용)
    nusc = None
    pred_boxes = None
    gt_boxes = None
    
    if args.scene_name or args.pred_boxes or args.gt_boxes or args.show_lidar:
        print("📊 NuScenes 데이터 로딩 중...")
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)
        config = config_factory('detection_cvpr_2019')
        
        # Load prediction boxes if provided
        if args.pred_boxes and os.path.exists(args.pred_boxes):
            print(f"📊 Prediction boxes 로딩 중: {args.pred_boxes}")
            pred_boxes, _ = load_prediction(args.pred_boxes, 
                                           config.max_boxes_per_sample, 
                                           DetectionBox,
                                           verbose=args.verbose)
            pred_boxes = add_ego_pose(nusc, pred_boxes)
        
        # Load ground truth boxes if provided
        if args.gt_boxes and os.path.exists(args.gt_boxes):
            print(f"📊 Ground truth boxes 로딩 중: {args.gt_boxes}")
            gt_boxes, _ = load_prediction(args.gt_boxes, 
                                         config.max_boxes_per_sample, 
                                         DetectionBox,
                                         verbose=args.verbose)
            gt_boxes = add_ego_pose(nusc, gt_boxes)
    
    # 애니메이션 생성
    if not os.path.exists(args.box_poses_dir):
        print(f"❌ Box poses 디렉토리를 찾을 수 없습니다: {args.box_poses_dir}")
        return
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.box_poses_dir, "box_optimization_animation")
    
    create_all_sample_animations(
        box_poses_dir=args.box_poses_dir,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        sample_token=args.sample_token,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        show_lidar=args.show_lidar,
        nusc=nusc,
        max_lidar_points=args.max_lidar_points
    )

if __name__ == "__main__":
    main()
    
