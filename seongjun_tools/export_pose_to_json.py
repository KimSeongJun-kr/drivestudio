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
    
    # yaw 회전 적용 (Z축 기준)
    angle_deg = 0
    angle_rad = np.deg2rad(angle_deg)
    yaw_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    world_rotation_matrix = yaw_rotation @ world_rotation_matrix
    
    world_rotation_quat = R.from_matrix(world_rotation_matrix).as_quat()  # [x, y, z, w]
    world_rotation_quat = np.array([world_rotation_quat[3], world_rotation_quat[0], world_rotation_quat[1], world_rotation_quat[2]])  # [w, x, y, z]
    
    return world_translation, world_rotation_quat

def extract_rigid_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: List[str] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
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

def extract_smpl_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: List[str] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
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

def extract_deformable_nodes_poses(checkpoint: Dict[str, Any], scene_token: str, scene_name: str, sample_tokens: List[str] = None, camera_front_start: Optional[np.ndarray] = None) -> List[PoseAnnotation]:
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

def extract_all_poses_from_checkpoint(checkpoint_path: str, output_dir: str, nuscenes_dataroot: Optional[str] = None, scene_name: str = "", nuscenes_version: str = "v1.0-mini") -> None:
    """checkpoint에서 모든 포즈 데이터를 추출하여 JSON으로 저장"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 로드된 데이터 타입 확인
    print(f"Loaded data type: {type(checkpoint)}")
    
    if not isinstance(checkpoint, dict):
        print(f"❌ Error: Expected dict, but got {type(checkpoint)}")
        print("Cannot process non-dictionary checkpoint files.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # NuScenes scene 정보와 sample 토큰 가져오기
    scene_token = ""
    sample_tokens = None
    camera_front_start = None
    
    if nuscenes_dataroot and scene_name:
        scene_token, scene_name, sample_tokens = get_nuscenes_scene_info(nuscenes_dataroot, scene_name, version=nuscenes_version)
        if sample_tokens:
            print(f"✅ NuScenes scene '{scene_name}' 정보 로드 완료")
            print(f"✅ NuScenes sample 토큰 로드 완료 ({len(sample_tokens)}개)")
            
            # 카메라 포즈 변환을 위한 첫 번째 카메라 포즈 로드
            camera_front_start = get_camera_front_start_pose(nuscenes_dataroot, scene_name, version=nuscenes_version)
            if camera_front_start is not None:
                print(f"✅ 카메라 front start 포즈 로드 완료 (좌표 변환용)")
            else:
                print(f"⚠️ 카메라 front start 포즈 로드 실패 - 좌표 변환 없이 진행")
        else:
            print(f"❌ 경로: '{nuscenes_dataroot}'에서 NuScenes scene '{scene_name}' 정보 로드 실패")
            return
    else:
        print(f"❌ 경로: '{nuscenes_dataroot}'에서 NuScenes scene '{scene_name}' 정보 로드 실패")
        return

    # 체크포인트 구조 분석
    print("\n🔍 Analyzing checkpoint structure...")
    analyze_checkpoint_structure(checkpoint)
    print()
        
    # 구조 검증
    print("🔍 Checking required structure...")
    structure_check = check_checkpoint_structure(checkpoint)
    
    for key, value in structure_check.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    if not structure_check['has_models']:
        print("❌ Error: No 'models' key found in checkpoint. Cannot extract poses.")
        return
    
    print()
       
    # 각 노드 타입별로 포즈 추출
    all_annotations = []
    
    if structure_check['has_rigid_nodes']:
        print("Extracting RigidNodes poses...")
        try:
            rigid_annotations = extract_rigid_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
            rigid_output = output_path / "rigid_nodes_poses.json"
            save_pose_annotations_to_json(rigid_annotations, str(rigid_output))
            print(f"✅ Saved {len(rigid_annotations)} RigidNodes annotations to {rigid_output}")
            all_annotations.extend(rigid_annotations)
        except Exception as e:
            print(f"❌ Error extracting RigidNodes: {e}")
    else:
        print("⏭️ Skipping RigidNodes (not found in checkpoint)")
    
    if structure_check['has_smpl_nodes']:
        print("Extracting SMPLNodes poses...")
        try:
            smpl_annotations = extract_smpl_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
            smpl_output = output_path / "smpl_nodes_poses.json"
            save_pose_annotations_to_json(smpl_annotations, str(smpl_output))
            print(f"✅ Saved {len(smpl_annotations)} SMPLNodes annotations to {smpl_output}")
            all_annotations.extend(smpl_annotations)
        except Exception as e:
            print(f"❌ Error extracting SMPLNodes: {e}")
    else:
        print("⏭️ Skipping SMPLNodes (not found in checkpoint)")
    
    if structure_check['has_deformable_nodes']:
        print("Extracting DeformableNodes poses...")
        try:
            deform_annotations = extract_deformable_nodes_poses(checkpoint, scene_token, scene_name, sample_tokens, camera_front_start)
            deform_output = output_path / "deformable_nodes_poses.json"
            save_pose_annotations_to_json(deform_annotations, str(deform_output))
            print(f"✅ Saved {len(deform_annotations)} DeformableNodes annotations to {deform_output}")
            all_annotations.extend(deform_annotations)
        except Exception as e:
            print(f"❌ Error extracting DeformableNodes: {e}")
    else:
        print("⏭️ Skipping DeformableNodes (not found in checkpoint)")
    
    # 추출된 포즈가 있으면 통합 파일 생성
    if all_annotations:
        print("Combining all poses...")
        all_output = output_path / "all_poses.json"
        save_pose_annotations_to_json(all_annotations, str(all_output))
        print(f"✅ Saved {len(all_annotations)} total annotations to {all_output}")
    else:
        print("⚠️ No pose data extracted from checkpoint")
    
    print("Pose extraction completed!")

def main():
    parser = argparse.ArgumentParser(description="Extract pose data from DriveStudio checkpoint")
    parser.add_argument("--checkpoint", type=str, default="/workspace/drivestudio/output/feasibility_check/run_original_scene_0_date_0529_try_1/checkpoint_final.pth", 
                       help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory for JSON files (default: checkpoint 경로의 keyframe_instance_poses_data 폴더)")
    parser.add_argument("--nuscenes-dataroot", type=str, default="/workspace/drivestudio/data/nuscenes/raw",
                       help="NuScenes 데이터 루트 디렉토리 (키프레임 매핑용)")
    parser.add_argument("--scene-name", type=str, default="scene-0061",
                       help="NuScenes scene 이름 (예: 'scene-0061')")
    parser.add_argument("--nuscenes-version", type=str, default="v1.0-mini",
                       help="NuScenes 버전 (예: 'v1.0-mini', 'v1.0-trainval')")
    
    args = parser.parse_args()
    
    # output이 지정되지 않은 경우 checkpoint 경로 기반으로 설정
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        args.output = str(checkpoint_path.parent / "keyframe_instance_poses_data")
    
    extract_all_poses_from_checkpoint(args.checkpoint, args.output, args.nuscenes_dataroot, args.scene_name, args.nuscenes_version)

if __name__ == "__main__":
    main()
