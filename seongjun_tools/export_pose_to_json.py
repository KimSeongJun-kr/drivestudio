import torch
import numpy as np
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import argparse
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import glob

@dataclass
class PoseAnnotation:
    """DriveStudio 포즈 데이터를 NuScenes annotation 포맷으로 저장하기 위한 클래스"""
    token: str                           # 고유 토큰
    sample_token: str                    # 샘플 토큰 (NuScenes 키프레임과 매칭)
    instance_token: str                  # 인스턴스 토큰
    visibility_token: str                # 가시성 토큰 (기본값 "4")
    attribute_tokens: List[str]          # 속성 토큰들
    translation: List[float]             # [x, y, z]
    size: List[float]                   # [w, l, h]
    rotation: List[float]               # quaternion [w, x, y, z]
    prev: str                           # 이전 annotation 토큰
    next: str                           # 다음 annotation 토큰
    num_lidar_pts: int                  # 라이다 포인트 수 (기본값 0)
    num_radar_pts: int                  # 레이더 포인트 수 (기본값 0)
    velocity: List[float]               # [vx, vy] 속도
    detection_name: str                 # 탐지 객체 이름
    detection_score: float              # 탐지 점수
    attribute_name: str                 # 속성 이름
    
    # DriveStudio 추가 정보
    instance_id: int                    # 인스턴스 ID (0~190)
    time_frame: int                     # 시간 프레임
    node_type: str                      # 'RigidNodes', 'SMPLNodes', 'DeformableNodes'
    confidence: float = 1.0             # 신뢰도
    scene_token: str = ""               # NuScenes scene 토큰
    scene_name: str = ""                # NuScenes scene 이름

def get_nuscenes_scene_info(dataroot: str, scene_name: str, version: str = 'v1.0-mini') -> Tuple[str, str, List[str]]:
    """NuScenes scene 이름으로 scene 정보와 sample 토큰들을 가져옴"""
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # scene 이름으로 scene 찾기
        scene = None
        for s in nusc.scene:
            if s['name'] == scene_name:
                scene = s
                break
                
        if scene is None:
            print(f"⚠️ Scene '{scene_name}'을 찾을 수 없습니다.")
            return "", "", []
            
        # scene의 첫 번째 샘플부터 시작
        sample = nusc.get('sample', scene['first_sample_token'])
        sample_tokens = []
        
        while True:
            sample_tokens.append(sample['token'])
            if sample['next'] == '':
                break
            sample = nusc.get('sample', sample['next'])
            
        return scene['token'], scene['name'], sample_tokens
        
    except Exception as e:
        print(f"⚠️ NuScenes scene 정보 가져오기 실패: {e}")
        return "", "", []

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

def extract_rigid_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: Optional[List[str]] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
    """RigidNodes 포즈 데이터 추출 (5의 배수 프레임만)"""
    rigid_quats = checkpoint['models']['RigidNodes']['instances_quats'].numpy() # (num_frames, num_instances, 4)
    rigid_trans = checkpoint['models']['RigidNodes']['instances_trans'].numpy() # (num_frames, num_instances, 3)
    rigid_sizes = checkpoint['models']['RigidNodes']['instances_size'].numpy()  # (num_instances, 3)
    
    annotations = []
    num_frames, num_instances = rigid_quats.shape[:2]
    
    # 5의 배수 프레임만 처리
    keyframe_indices = [i for i in range(0, num_frames, 5)]
    
    # sample_tokens가 제공된 경우 개수 확인
    if sample_tokens and len(sample_tokens) != len(keyframe_indices):
        print(f"⚠️ 경고: NuScenes sample 수({len(sample_tokens)})와 키프레임 수({len(keyframe_indices)})가 일치하지 않습니다.")
    
    for instance_id in range(num_instances):
        instance_size = rigid_sizes[instance_id].tolist()
        instance_size = [instance_size[1], instance_size[0], instance_size[2]]
        
        for idx, frame_id in enumerate(keyframe_indices):
            sample_token = sample_tokens[idx] if sample_tokens and idx < len(sample_tokens) else ""
            
            translation = rigid_trans[frame_id, instance_id]
            rotation = rigid_quats[frame_id, instance_id]
            
            # translation이 [0, 0, 0]인 경우 건너뛰기
            if np.allclose(translation, [0, 0, 0], atol=1e-6):
                continue
            
            # camera 좌표계에서 world 좌표계로 변환
            if camera_front_start is not None:
                translation, rotation = transform_pose_to_world(translation, rotation, camera_front_start)
            
            translation = translation.tolist()
            rotation = rotation.tolist()
            
            annotation = PoseAnnotation(
                token="",
                sample_token=sample_token,
                instance_token="",
                visibility_token="",
                attribute_tokens=[],
                translation=translation,
                size=instance_size,
                rotation=rotation,
                prev="",
                next="",
                num_lidar_pts=0,
                num_radar_pts=0,
                velocity=[0.0, 0.0],
                detection_name="car",
                detection_score=1.0,
                attribute_name="vehicle.moving",
                instance_id=instance_id,
                time_frame=frame_id,
                node_type='RigidNodes',
                confidence=1.0,
                scene_token=scene_token,
                scene_name=scene_name
            )
            annotations.append(annotation)
    
    return annotations

def extract_smpl_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: Optional[List[str]] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
    """SMPLNodes 포즈 데이터 추출 (5의 배수 프레임만)"""
    smpl_instance_quats = checkpoint['models']['SMPLNodes']['instances_quats'].numpy()  # (num_frames, num_instances, 1, 4)
    smpl_instance_trans = checkpoint['models']['SMPLNodes']['instances_trans'].numpy()  # (num_frames, num_instances, 3)
    smpl_sizes = checkpoint['models']['SMPLNodes']['instances_size'].numpy()         # (num_instances, 3)
    
    annotations = []
    num_frames, num_instances = smpl_instance_trans.shape[:2]
    
    # 5의 배수 프레임만 처리
    keyframe_indices = [i for i in range(0, num_frames, 5)]
    
    # sample_tokens가 제공된 경우 개수 확인
    if sample_tokens and len(sample_tokens) != len(keyframe_indices):
        print(f"⚠️ 경고: NuScenes sample 수({len(sample_tokens)})와 키프레임 수({len(keyframe_indices)})가 일치하지 않습니다.")
    
    for instance_id in range(num_instances):
        instance_size = smpl_sizes[instance_id].tolist()
        instance_size = [instance_size[1], instance_size[0], instance_size[2]]

        for idx, frame_id in enumerate(keyframe_indices):
            sample_token = sample_tokens[idx] if sample_tokens and idx < len(sample_tokens) else ""
            
            translation = smpl_instance_trans[frame_id, instance_id]
            rotation = smpl_instance_quats[frame_id, instance_id, 0]  # (1, 4) -> (4,)
            
            # translation이 [0, 0, 0]인 경우 건너뛰기
            if np.allclose(translation, [0, 0, 0], atol=1e-6):
                continue
            
            # camera 좌표계에서 world 좌표계로 변환
            if camera_front_start is not None:
                translation, rotation = transform_pose_to_world(translation, rotation, camera_front_start)
            
            translation = translation.tolist()
            rotation = rotation.tolist()
            
            annotation = PoseAnnotation(
                token="",
                sample_token=sample_token,
                instance_token="",
                visibility_token="",
                attribute_tokens=[],  # SMPL 관련 속성
                translation=translation,
                size=instance_size,
                rotation=rotation,
                prev="",
                next="",
                num_lidar_pts=0,
                num_radar_pts=0,
                velocity=[0.0, 0.0],
                detection_name="pedestrian",
                detection_score=1.0,
                attribute_name="pedestrian.moving",
                instance_id=instance_id,
                time_frame=frame_id,
                node_type='SMPLNodes',
                confidence=1.0,
                scene_token=scene_token,
                scene_name=scene_name
            )
            annotations.append(annotation)
    
    return annotations

def extract_deformable_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: Optional[List[str]] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
    """DeformableNodes 포즈 데이터 추출 (5의 배수 프레임만)"""
    deform_quats = checkpoint['models']['DeformableNodes']['instances_quats'].numpy()  # (num_frames, num_instances, 4)
    deform_trans = checkpoint['models']['DeformableNodes']['instances_trans'].numpy()  # (num_frames, num_instances, 3)
    deform_sizes = checkpoint['models']['DeformableNodes']['instances_size'].numpy()  # (num_instances, 3)
    
    annotations = []
    num_frames, num_instances = deform_quats.shape[:2]
    
    # 5의 배수 프레임만 처리
    keyframe_indices = [i for i in range(0, num_frames, 5)]
    
    # sample_tokens가 제공된 경우 개수 확인
    if sample_tokens and len(sample_tokens) != len(keyframe_indices):
        print(f"⚠️ 경고: NuScenes sample 수({len(sample_tokens)})와 키프레임 수({len(keyframe_indices)})가 일치하지 않습니다.")
    
    for instance_id in range(num_instances):
        # 인스턴스별 크기 정보
        instance_size = deform_sizes[instance_id].tolist()
        instance_size = [instance_size[1], instance_size[0], instance_size[2]]

        for idx, frame_id in enumerate(keyframe_indices):
            sample_token = sample_tokens[idx] if sample_tokens and idx < len(sample_tokens) else ""
            
            translation = deform_trans[frame_id, instance_id]
            rotation = deform_quats[frame_id, instance_id]
            
            # translation이 [0, 0, 0]인 경우 건너뛰기
            if np.allclose(translation, [0, 0, 0], atol=1e-6):
                continue
            
            # camera 좌표계에서 world 좌표계로 변환
            if camera_front_start is not None:
                translation, rotation = transform_pose_to_world(translation, rotation, camera_front_start)
            
            translation = translation.tolist()
            rotation = rotation.tolist()
            
            annotation = PoseAnnotation(
                token="",
                sample_token=sample_token,
                instance_token="",
                visibility_token="",
                attribute_tokens=[],
                translation=translation,
                size=instance_size,
                rotation=rotation,
                prev="",
                next="",
                num_lidar_pts=0,
                num_radar_pts=0,
                velocity=[0.0, 0.0],
                detection_name="bicycle",
                detection_score=1.0,
                attribute_name="cycle.with_rider",
                instance_id=instance_id,
                time_frame=frame_id,
                node_type='DeformableNodes',
                confidence=1.0,
                scene_token=scene_token,
                scene_name=scene_name
            )
            annotations.append(annotation)
    
    return annotations

def pose_annotation_to_dict(annotation: PoseAnnotation) -> OrderedDict:
    """PoseAnnotation을 OrderedDict로 변환 (NuScenes 포맷)"""
    result = OrderedDict([
        ('token', annotation.token),
        ('sample_token', annotation.sample_token),
        ('instance_token', annotation.instance_token),
        ('visibility_token', annotation.visibility_token),
        ('attribute_tokens', annotation.attribute_tokens),
        ('translation', annotation.translation),
        ('size', annotation.size),
        ('rotation', annotation.rotation),
        ('prev', annotation.prev),
        ('next', annotation.next),
        ('num_lidar_pts', annotation.num_lidar_pts),
        ('num_radar_pts', annotation.num_radar_pts),
        ('velocity', annotation.velocity),
        ('detection_name', annotation.detection_name),
        ('detection_score', annotation.detection_score),
        ('attribute_name', annotation.attribute_name),
        # DriveStudio 추가 필드
        ('instance_id', annotation.instance_id),
        ('time_frame', annotation.time_frame),
        ('node_type', annotation.node_type),
        ('confidence', annotation.confidence),
        ('scene_token', annotation.scene_token),
        ('scene_name', annotation.scene_name)
    ])
    
    return result

def save_pose_annotations_to_json(annotations: List[PoseAnnotation], output_path: str) -> None:
    """포즈 annotation을 JSON 파일로 저장 (sample_token별로 그룹화)"""
    # sample_token별로 그룹화
    results = {}
    for annotation in annotations:
        sample_token = annotation.sample_token
        if sample_token not in results:
            results[sample_token] = []
        
        # 딕셔너리로 변환하여 추가
        annotation_dict = pose_annotation_to_dict(annotation)
        results[sample_token].append(annotation_dict)
    
    # 최종 JSON 구조 생성
    output_data = {
        "meta": {
            "use_camera": False,
            "use_lidar": True
        },
        "results": results
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def analyze_checkpoint_structure(checkpoint: Dict[str, Any]) -> None:
    """체크포인트의 구조를 간단히 분석하고 출력"""
    print("=" * 50)
    print("CHECKPOINT STRUCTURE")
    print("=" * 50)
    
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{key}/ (dict with {len(value)} keys)")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"  {sub_key}/ (dict with {len(sub_value)} keys)")
                    for deep_key, deep_value in sub_value.items():
                        if hasattr(deep_value, 'shape'):
                            print(f"    {deep_key}: {list(deep_value.shape)} ({deep_value.dtype})")
                        else:
                            print(f"    {deep_key}: {type(deep_value).__name__}")
                elif hasattr(sub_value, 'shape'):
                    print(f"  {sub_key}: {list(sub_value.shape)} ({sub_value.dtype})")
                else:
                    print(f"  {sub_key}: {type(sub_value).__name__}")
        elif hasattr(value, 'shape'):
            print(f"{key}: {list(value.shape)} ({value.dtype})")
        else:
            print(f"{key}: {type(value).__name__}")
    print("=" * 50)

def check_checkpoint_structure(checkpoint: Dict[str, Any]) -> Dict[str, bool]:
    """체크포인트가 예상한 구조를 가지고 있는지 확인"""
    structure_check = {
        'has_models': False,
        'has_rigid_nodes': False,
        'has_smpl_nodes': False,
        'has_deformable_nodes': False
    }
    
    try:
        if 'models' in checkpoint:
            structure_check['has_models'] = True
            models = checkpoint['models']
            
            # RigidNodes 확인
            if ('RigidNodes' in models and 
                'instances_quats' in models['RigidNodes'] and 
                'instances_trans' in models['RigidNodes'] and
                'instances_size' in models['RigidNodes']):
                structure_check['has_rigid_nodes'] = True
            
            # SMPLNodes 확인
            if ('SMPLNodes' in models and 
                'instances_quats' in models['SMPLNodes'] and 
                'instances_trans' in models['SMPLNodes'] and
                'instances_size' in models['SMPLNodes']):
                structure_check['has_smpl_nodes'] = True
            
            # DeformableNodes 확인
            if ('DeformableNodes' in models and 
                'instances_quats' in models['DeformableNodes'] and 
                'instances_trans' in models['DeformableNodes'] and
                'instances_size' in models['DeformableNodes']):
                structure_check['has_deformable_nodes'] = True
                
    except Exception as e:
        print(f"Error checking structure: {e}")
    
    return structure_check

def process_folder_checkpoints(folder_path: str, checkpoint_filename: str = "checkpoint_final.pth", analyze_structure: bool = False, nuscenes_dataroot: Optional[str] = None, nuscenes_version: str = "v1.0-mini") -> None:
    """폴더 내 모든 하위 폴더의 checkpoint 파일들을 일괄 처리하여 하나의 JSON 파일로 통합"""
    # scene 이름 매핑 리스트
    scene_names = ['scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', 'scene-0757', 
                   'scene-0796', 'scene-0916', 'scene-1077', 'scene-1094', 'scene-1100']
    
    folder_path_obj = Path(folder_path)
    
    if not folder_path_obj.exists():
        print(f"❌ 오류: 폴더 '{folder_path}'가 존재하지 않습니다.")
        return
    
    if not folder_path_obj.is_dir():
        print(f"❌ 오류: '{folder_path}'는 폴더가 아닙니다.")
        return
    
    print(f"🔍 폴더 '{folder_path}' 내에서 {checkpoint_filename} 파일들을 검색중...")
    
    # 하위 폴더들에서 checkpoint 파일 찾기
    checkpoint_pattern = str(folder_path_obj / "*" / checkpoint_filename)
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"❌ '{folder_path}' 내에서 {checkpoint_filename} 파일을 찾을 수 없습니다.")
        return
    
    print(f"✅ 총 {len(checkpoint_files)}개의 {checkpoint_filename} 파일을 찾았습니다:")
    for checkpoint_file in checkpoint_files:
        print(f"  - {checkpoint_file}")
    
    # 모든 checkpoint에서 추출한 annotation들을 저장할 리스트
    all_combined_annotations = []
    
    # 각 checkpoint 파일 처리
    for i, checkpoint_file in enumerate(checkpoint_files):
        checkpoint_path = Path(checkpoint_file)
        parent_folder = checkpoint_path.parent
        folder_name = parent_folder.name
        
        print(f"\n{'='*80}")
        print(f"처리 중 ({i+1}/{len(checkpoint_files)}): {folder_name}")
        print(f"{'='*80}")
        
        # 폴더명에서 scene 인덱스 추출
        import re
        scene_match = re.search(r'scene_(\d+)', folder_name)
        if scene_match:
            scene_index = int(scene_match.group(1))
            if scene_index < len(scene_names):
                scene_name = scene_names[scene_index]
                print(f"📍 폴더 '{folder_name}'에서 scene 인덱스 {scene_index} 추출 → '{scene_name}'")
            else:
                print(f"⚠️ scene 인덱스 {scene_index}가 scene_names 리스트 범위를 초과합니다. 기본값 사용.")
                scene_name = scene_names[0]  # 첫 번째 scene을 기본값으로 사용
        else:
            print(f"⚠️ 폴더명 '{folder_name}'에서 scene 정보를 추출할 수 없습니다. 기본값 사용.")
            scene_name = scene_names[0]  # 첫 번째 scene을 기본값으로 사용
        
        # NuScenes scene 정보와 sample 토큰 가져오기
        scene_token = ""
        sample_tokens = None
        camera_front_start = None
        
        if nuscenes_dataroot and scene_name:
            scene_token, scene_name, sample_tokens = get_nuscenes_scene_info(nuscenes_dataroot, scene_name, version=nuscenes_version)
            if sample_tokens:
                print(f"✅ NuScenes scene '{scene_name}' 정보 로드 완료 ({len(sample_tokens)}개 sample)")
                
                # 카메라 포즈 변환을 위한 첫 번째 카메라 포즈 로드
                camera_front_start = get_camera_front_start_pose(nuscenes_dataroot, scene_name, version=nuscenes_version)
                if camera_front_start is not None:
                    print(f"✅ 카메라 front start 포즈 로드 완료")
                else:
                    print(f"⚠️ 카메라 front start 포즈 로드 실패 - 좌표 변환 없이 진행")
            else:
                print(f"❌ NuScenes scene '{scene_name}' 정보 로드 실패 - 이 checkpoint 건너뛰기")
                continue
        else:
            print(f"❌ NuScenes 정보가 제공되지 않았습니다 - 이 checkpoint 건너뛰기")
            continue
        
        try:
            # checkpoint 로드
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            # 로드된 데이터 타입 확인
            if not isinstance(checkpoint, dict):
                print(f"❌ Error: Expected dict, but got {type(checkpoint)}")
                continue
            
            # 체크포인트 구조 분석 (옵션)
            if analyze_structure:
                print("\n🔍 Analyzing checkpoint structure...")
                analyze_checkpoint_structure(checkpoint)
                print()
            
            # 구조 검증
            structure_check = check_checkpoint_structure(checkpoint)
            
            if not structure_check['has_models']:
                print("❌ Error: No 'models' key found in checkpoint. Skipping.")
                continue
            
            # 각 노드 타입별로 포즈 추출
            checkpoint_annotations = []
            
            if structure_check['has_rigid_nodes']:
                print("Extracting RigidNodes poses...")
                rigid_annotations = extract_rigid_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
                checkpoint_annotations.extend(rigid_annotations)
                print(f"✅ Extracted {len(rigid_annotations)} RigidNodes annotations")
            
            if structure_check['has_smpl_nodes']:
                print("Extracting SMPLNodes poses...")
                smpl_annotations = extract_smpl_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
                checkpoint_annotations.extend(smpl_annotations)
                print(f"✅ Extracted {len(smpl_annotations)} SMPLNodes annotations")
            
            if structure_check['has_deformable_nodes']:
                print("Extracting DeformableNodes poses...")
                deform_annotations = extract_deformable_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
                checkpoint_annotations.extend(deform_annotations)
                print(f"✅ Extracted {len(deform_annotations)} DeformableNodes annotations")
            
            # 현재 checkpoint의 annotation들을 전체 리스트에 추가
            all_combined_annotations.extend(checkpoint_annotations)
            print(f"✅ 완료: {folder_name} ({len(checkpoint_annotations)} annotations)")
            
        except Exception as e:
            print(f"❌ 오류 발생 ({folder_name}): {e}")
            continue
    
    # 모든 결과를 하나의 JSON 파일로 저장
    if all_combined_annotations:
        output_file = folder_path_obj / "poses.json"
        save_pose_annotations_to_json(all_combined_annotations, str(output_file))
        print(f"\n🎉 모든 checkpoint 파일 처리 완료!")
        print(f"✅ 총 {len(all_combined_annotations)}개의 annotation을 '{output_file}'에 저장했습니다.")
        print(f"처리된 checkpoint 파일: {len(checkpoint_files)}개")
    else:
        print(f"\n⚠️ 추출된 포즈 데이터가 없습니다.")

def main():
    parser = argparse.ArgumentParser(description="Extract pose data from DriveStudio checkpoints")
    
    # 폴더 일괄 처리 옵션
    parser.add_argument("--folder", type=str,
                        default="/workspace/drivestudio/output/feasibility_check/updated",
                        help="Path to folder containing subfolders with checkpoint files")
    parser.add_argument("--checkpoint-filename", type=str, 
                       default="checkpoint_final.pth",
                       help="Name of checkpoint file to search for (default: checkpoint_final.pth)")
    parser.add_argument("--analyze-structure", type=bool, 
                       default=False,
                       help="Analyze and print checkpoint structure for each file")
    
    # 공통 옵션들
    parser.add_argument("--nuscenes-dataroot", type=str, 
                        default="/workspace/drivestudio/data/nuscenes/raw",
                        help="NuScenes 데이터 루트 디렉토리 (키프레임 매핑용)")
    parser.add_argument("--nuscenes-version", type=str, 
                        default="v1.0-mini",
                        help="NuScenes 버전 (예: 'v1.0-mini', 'v1.0-trainval')")
    
    args = parser.parse_args()
    
    # 폴더 일괄 처리
    print(f"🚀 폴더 일괄 처리 모드")
    print(f"📁 대상 폴더: {args.folder}")
    print(f"📄 검색할 파일명: {args.checkpoint_filename}")
    if args.analyze_structure:
        print(f"🔍 체크포인트 구조 분석: 활성화")
    process_folder_checkpoints(
        folder_path=args.folder,
        checkpoint_filename=args.checkpoint_filename,
        analyze_structure=args.analyze_structure,
        nuscenes_dataroot=args.nuscenes_dataroot,
        nuscenes_version=args.nuscenes_version
    )

if __name__ == "__main__":
    main()
