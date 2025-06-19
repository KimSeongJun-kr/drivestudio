import argparse
import copy
import numpy as np
import tqdm
from typing import Callable, Tuple, List, Dict
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionBox, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, _get_box_class_field
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

@dataclass
class Annotation:
    token: str
    sample_token: str
    instance_token: str
    visibility_token: str
    attribute_tokens: List[str]
    translation: List[float]     # [x, y, z]
    size: List[float]            # [w, l, h]
    rotation: List[float]        # quaternion [w, x, y, z]
    prev: str
    next: str
    num_lidar_pts: int
    num_radar_pts: int

def correspondence(tar_boxes: EvalBoxes,
                src_boxes: EvalBoxes,
                dist_fcn: Callable,
                dist_th: float,
                verbose: bool = False) -> Tuple[Dict[str, List[DetectionBox]], Dict[str, List[DetectionBox]]]:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param tar_boxes: Maps every sample_token to a list of its sample_annotations.
    :param src_boxes: Maps every sample_token to a list of its sample_results.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: Dictionary mapping sample_token to list of matched source boxes.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for tar_box in tar_boxes.all])
    if verbose:
        print("Found {} GT of all classes out of {} total across {} samples.".
              format(npos, len(tar_boxes.all), len(tar_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return {}, {}

    # Organize the predictions in a single list.
    src_boxes_list = [box for box in src_boxes.all]
    src_confs = [box.detection_score for box in src_boxes_list]

    if verbose:
        print("Found {} PRED of all class out of {} total across {} samples.".
              format(len(src_confs), len(src_boxes.all), len(src_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(src_confs))][::-1]

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    match_src_boxes = defaultdict(list)
    match_tar_boxes = defaultdict(list)
    for ind in sortind:
        src_box = src_boxes_list[ind]
        min_dist = np.inf
        match_tar_idx = None

        # Check if the sample_token exists in tar_boxes
        if src_box.sample_token not in tar_boxes.sample_tokens:
            continue

        for tar_idx, tar_box in enumerate(tar_boxes[src_box.sample_token]):

            # Find closest match among ground truth boxes
            if not (src_box.sample_token, tar_idx) in taken:
                this_distance = dist_fcn(tar_box, src_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_tar_idx = tar_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th and match_tar_idx is not None

        if is_match:
            assert match_tar_idx is not None  # Type assertion for linter
            taken.add((src_box.sample_token, match_tar_idx))
            match_src_boxes[src_box.sample_token].append(src_box)
            match_tar_boxes[src_box.sample_token].append(tar_boxes[src_box.sample_token][match_tar_idx])

    return match_src_boxes, match_tar_boxes

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

def get_scenes_from_boxes(nusc: NuScenes, boxes: EvalBoxes) -> List[str]:
    """EvalBoxes에서 scene 이름들을 추출합니다.

    Args:
        nusc: NuScenes 객체
        boxes: scene 이름을 추출할 EvalBoxes

    Returns:
        scene 이름들의 리스트
    """
    scene_names = set()
    for sample_token in boxes.sample_tokens:
        sample = nusc.get('sample', sample_token)
        scene_token = sample['scene_token']
        scene = nusc.get('scene', scene_token)
        scene_names.add(scene['name'])
    
    return list(scene_names)

def filter_boxes_by_common_scenes(nusc: NuScenes, src_boxes: EvalBoxes, tar_boxes: EvalBoxes) -> Tuple[EvalBoxes, EvalBoxes]:
    """src와 tar에 모두 포함된 scene들만으로 boxes를 필터링합니다.

    Args:
        nusc: NuScenes 객체
        src_boxes: source EvalBoxes
        tar_boxes: target EvalBoxes

    Returns:
        필터링된 (src_boxes, tar_boxes) 튜플
    """
    # 각각에서 scene들 추출
    src_scenes = set(get_scenes_from_boxes(nusc, src_boxes))
    tar_scenes = set(get_scenes_from_boxes(nusc, tar_boxes))
    
    # 공통 scene들 찾기
    common_scenes = src_scenes.intersection(tar_scenes)
    
    print(f"📊 Source scenes: {len(src_scenes)}, Target scenes: {len(tar_scenes)}, Common scenes: {len(common_scenes)}")
    print(f"🔄 Common scenes: {sorted(common_scenes)}")
    
    if not common_scenes:
        print("⚠️ 공통 scene이 없습니다!")
        return EvalBoxes(), EvalBoxes()
    
    # scene token들로 변환
    common_scene_tokens = set()
    for scene in nusc.scene:
        if scene['name'] in common_scenes:
            common_scene_tokens.add(scene['token'])
    
    # 공통 scene들에 해당하는 sample_tokens만 필터링
    def filter_by_scene_tokens(boxes: EvalBoxes, scene_tokens: set) -> EvalBoxes:
        filtered_boxes = EvalBoxes()
        for sample_token in boxes.sample_tokens:
            sample = nusc.get('sample', sample_token)
            if sample['scene_token'] in scene_tokens:
                filtered_boxes.add_boxes(sample_token, boxes[sample_token])
        return filtered_boxes
    
    filtered_src_boxes = filter_by_scene_tokens(src_boxes, common_scene_tokens)
    filtered_tar_boxes = filter_by_scene_tokens(tar_boxes, common_scene_tokens)
    
    print(f"✅ 필터링 완료: src samples: {len(filtered_src_boxes.sample_tokens)}, tar samples: {len(filtered_tar_boxes.sample_tokens)}")
    
    return filtered_src_boxes, filtered_tar_boxes

def write_prediction_file(box_list: Dict[str, List[DetectionBox]], output_path: str) -> None:
    """DetectionBox 객체 리스트를 NuScenes prediction 포맷의 JSON 파일로 저장합니다.
    
    Args:
        box_list: sample_token별로 그룹화된 DetectionBox 리스트
        output_path: 저장할 파일 경로
    """
    # 기본 meta 정보
    meta = {
        "use_camera": False,
        "use_lidar": True
    }
    
    # sample_token별로 그룹화
    results = defaultdict(list)
    
    for sample_token, boxes in box_list.items():
        for box in boxes:
            # attribute_name 설정 - 실제 박스의 값 사용
            attribute_name = box.attribute_name if box.attribute_name else ""
                
            # detection_name 설정 - 실제 박스의 값 사용
            detection_name = box.detection_name
            
            # detection_score 설정 - 실제 박스의 값 사용, 음수인 경우 기본값 사용
            detection_score = box.detection_score if box.detection_score >= 0 else 0.5
            
            prediction = OrderedDict([
                ("sample_token", box.sample_token),
                ("translation", [float(x) for x in box.translation]),
                ("size", [float(x) for x in box.size]),
                ("rotation", [float(x) for x in box.rotation]),
                ("velocity", [float(x) for x in box.velocity]),  # 실제 velocity 값 사용
                ("detection_name", detection_name),
                ("detection_score", detection_score),
                ("attribute_name", attribute_name)
            ])
            
            results[box.sample_token].append(prediction)
    
    # 최종 prediction 파일 구조
    prediction_data = {
        "meta": meta,
        "results": dict(results)
    }
    
    with open(output_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--src",
        type=str,
        # default="/workspace/drivestudio/output/feasibility_check/updated/poses.json",
        default="/workspace/drivestudio/output/feasibility_check/updated/poses_selected_tar_selected_src.json",
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred.json",
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_gt_pred.json",
        help="Path to source prediction json",
    )
    parser.add_argument(
        "--tar",
        type=str,
        # default="/workspace/drivestudio/output/feasibility_check/updated/poses.json",
        # default="/workspace/drivestudio/output/feasibility_check/updated/poses_selected_tar.json",        
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred.json",
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc.json",
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_selected_tar.json",
        default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_gt_pred_selected_src.json",
        help="Path to destination gaussian poses json",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default="",
        # default="_matched",
        help="Postfix for output file name",
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
        "--scene_name",
        type=str,
        default=None,
        help="Scene name to filter boxes (e.g., 'scene-0061')",
    )

    args = parser.parse_args()

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
    
    assert os.path.exists(args.src), 'Error: The result file does not exist!'
    assert os.path.exists(args.tar), 'Error: The result file does not exist!'
    config = config_factory('detection_cvpr_2019')

    print('Loading src prediction...')
    src_pred_boxes, src_meta = load_prediction(args.src, 
                                        config.max_boxes_per_sample, 
                                        DetectionBox,
                                        verbose=args.verbose)
    print('Loading tar prediction...')
    tar_pred_boxes, tar_meta = load_prediction(args.tar, 
                                        config.max_boxes_per_sample, 
                                        DetectionBox,
                                        verbose=args.verbose)


    # Filter boxes by scene if scene_name is provided
    if args.scene_name:
        if args.verbose:
            print(f"Filtering boxes by scene: {args.scene_name}")
        src_pred_boxes = filter_boxes_by_scene(nusc, src_pred_boxes, args.scene_name)
        tar_pred_boxes = filter_boxes_by_scene(nusc, tar_pred_boxes, args.scene_name)
    else:
        # Filter to only include scenes that exist in both src and tar
        if args.verbose:
            print("Filtering to common scenes in both src and tar...")
        src_pred_boxes, tar_pred_boxes = filter_boxes_by_common_scenes(nusc, src_pred_boxes, tar_pred_boxes)

    # Note: We no longer require exact sample token matches since we're filtering by common scenes
    # but we can still check if there are any overlapping samples
    common_samples = set(src_pred_boxes.sample_tokens).intersection(set(tar_pred_boxes.sample_tokens))
    print(f"📊 Common samples between src and tar: {len(common_samples)}")
    print(f"🔄 Total src samples: {len(src_pred_boxes.sample_tokens)}, Total tar samples: {len(tar_pred_boxes.sample_tokens)}")
    
    print('Matching boxes...')
    all_matched_src_boxes = defaultdict(list)
    all_matched_tar_boxes = defaultdict(list)
    dist_th = 2.0

    match_src_boxes, match_tar_boxes = correspondence(tar_pred_boxes, src_pred_boxes, config.dist_fcn_callable, dist_th)
    for sample_token, matched_src_boxes in match_src_boxes.items():
        all_matched_src_boxes[sample_token].extend(matched_src_boxes)
    for sample_token, matched_tar_boxes in match_tar_boxes.items():
        all_matched_tar_boxes[sample_token].extend(matched_tar_boxes)
    selected_src_path = args.src.replace('.json', f'{args.output_postfix}_selected_src.json')
    selected_tar_path = args.tar.replace('.json', f'_selected_tar.json')
    
    write_prediction_file(all_matched_src_boxes, selected_src_path)
    write_prediction_file(all_matched_tar_boxes, selected_tar_path)
    
    print(f"✅ Selected instances are saved to {selected_src_path} and {selected_tar_path}")
    
    # 전체 매칭된 박스의 개수 계산
    total_matched_boxes = sum(len(boxes) for boxes in all_matched_src_boxes.values())
    print(f"num src_pred: {len(src_pred_boxes.all)}, num tar_pred: {len(tar_pred_boxes.all)}, num matched: {total_matched_boxes}")

if __name__ == "__main__":
    main()
    
