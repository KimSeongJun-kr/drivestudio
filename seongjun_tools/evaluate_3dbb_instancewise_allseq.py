import argparse
import copy
import numpy as np
import tqdm
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import glob
import pandas as pd
from pathlib import Path
import re
from typing import Callable, Tuple

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.algo import calc_ap
from nuscenes.eval.detection.constants import TP_METRICS
# from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean



# Import functions from other modules
import sys
sys.path.append('/workspace/drivestudio')
from seongjun_tools.generate_selected_instances import filter_boxes_by_common_scenes
from seongjun_tools.evaluate_3dbb import filter_eval_boxes, filter_boxes_by_scene
from seongjun_tools.utils.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics, DetectionMetricData
from seongjun_tools.utils.loaders import load_gt, load_prediction, add_center_dist
from seongjun_tools.utils.splits import create_splits_scenes
from seongjun_tools.evaluate_3dbb_instancewise import perform_evaluation, match_boxes, _get_scene_sample_tokens_chronologically, extract_all_boxes_from_json, find_files_with_name, parse_metrics_json
from datasets.nuscenes.nuscenes_sourceloader import OBJECT_CLASS_NODE_MAPPING
from datasets.base.scene_dataset import ModelType

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

def extract_boxes_from_json_to_evalboxes_multiple(nusc, target_file_scene_pairs: List[Tuple[str, str]]) -> EvalBoxes:
    """JSON 파일에서 박스 정보를 추출하여 EvalBoxes 형태로 변환합니다.
    
    Args:
        json_path: JSON 파일 경로
        multi_scenes: 사용할 scene 리스트
        
    Returns:
        EvalBoxes 객체
    """
    eval_boxes = EvalBoxes()

    for target_file, scene_name in target_file_scene_pairs:
        frame_boxes = extract_all_boxes_from_json(target_file)
        if frame_boxes is None:
            print(f"❌ JSON 파일에서 박스 정보를 추출할 수 없습니다: {target_file}")
            return EvalBoxes()
        
        sample_tokens = _get_scene_sample_tokens_chronologically(nusc, scene_name)

        # sample_tokens와 frame_id 매핑 (순서대로 매칭)
        for kf_id, sample_token in enumerate(sample_tokens):
            f_id = kf_id * 5
            if f_id in frame_boxes:
                boxes = frame_boxes[f_id]
                detection_boxes = []
                
                for box in boxes:
                    try:
                        # box가 딕셔너리인지 확인
                        if not isinstance(box, dict):
                            print(f"❌ 박스 정보가 딕셔너리가 아닙니다: {type(box)}")
                            continue
                        detection_name = box.get('detection_name', '')
                        if detection_name == 'human.pedestrian.personal_mobility':
                            continue

                        # box 정보를 DetectionBox로 변환
                        detection_box = DetectionBox(
                            sample_token=sample_token,
                            translation=box.get('translation', [0, 0, 0]),
                            size=box.get('size', [1, 1, 1]),
                            rotation=box.get('rotation', [1, 0, 0, 0]),
                            velocity=box.get('velocity', [0, 0]),
                            detection_name=detection_name,
                            detection_score=box.get('detection_score', 0.5),
                            attribute_name=box.get('attribute_name', ''),
                            instance_token=box.get('instance_token', ''),
                            instance_idx=box.get('instance_idx', -1),
                            num_gaussians=box.get('num_gaussians', -1)
                        )
                        detection_boxes.append(detection_box)
                    except Exception as e:
                        print(f"❌ 박스 변환 중 오류 발생: {e}")
                        print(f"   박스 정보: {box}")
                        continue
                
                if detection_boxes:
                    eval_boxes.add_boxes(sample_token, detection_boxes)
    
    return eval_boxes

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-file 3D bounding box evaluation")
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Path to ground truth prediction json file",
    )
    parser.add_argument(
        "--ctrl",
        type=str,
        default='/workspace/drivestudio/data/nuscenes/raw/v1.0-mini/boxes_noise_bias.json',
        # default='/workspace/drivestudio/data/nuscenes/raw/v1.0-mini/boxes_centerpoint.json',
        help="Path to comparison prediction json file",
    )
    parser.add_argument(
        "--tar",
        type=str,
        default='/workspace/drivestudio/output/box_experiments_0910',
        help="Directory to search for target files",
    )
    parser.add_argument(
        "--name",
        type=str,
        default='box_poses_50000.json',
        help="Name of the files to find in tar directory",
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
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Verbose",
    )
    args = parser.parse_args()

    # Verify input files exist
    if args.gt is not None:
        assert os.path.exists(args.gt), f'Error: GT file does not exist: {args.gt}'
    if args.ctrl is not None:
        assert os.path.exists(args.ctrl), f'Error: Compare file does not exist: {args.ctrl}'
    if args.tar is not None:
        assert os.path.exists(args.tar), f'Error: Tar directory does not exist: {args.tar}'

    scene_splits = create_splits_scenes()

    eval_set_map = {
        'v1.0-mini': 'mini_trainval',
        'v1.0-trainval': 'val',
    }

    mini_scenes = scene_splits[eval_set_map[args.version]]

    # Find all target files
    print(f'🔍 Finding {args.name} files in {args.tar}...')
    target_files = find_files_with_name(args.tar, args.name)
    
    if not target_files:
        print(f"❌ No {args.name} files found in {args.tar}")
        return

    # target_files 각 요소의 경로에서 부모 디렉토리 중 *_seq<number> 패턴의 마지막 숫자(씬 인덱스)를 추출합니다.
    file_seq_pairs = []
    for target_file in target_files:
        seq_index = None
        for part in reversed(Path(target_file).parts):
            m = re.search(r"_seq(\d+)$", part)
            if m is not None:
                seq_index = int(m.group(1))
                break
        if seq_index is not None:
            file_seq_pairs.append((target_file, seq_index))
    tar_scene_indices = sorted({seq for _, seq in file_seq_pairs})
    if args.verbose:
        print(f"🎯 사용될 scene 인덱스 추출: {tar_scene_indices}")

    # (파일경로, scene_name) 튜플 리스트 생성 및 파생 리스트 구성
    target_file_scene_pairs = [
        (file_path, mini_scenes[idx])
        for file_path, idx in file_seq_pairs
        if 0 <= idx < len(mini_scenes)
    ]
    filtered_target_files = [file_path for file_path, _ in target_file_scene_pairs]
    target_scenes = [scene for _, scene in target_file_scene_pairs]
    if args.verbose:
        print(f"🎯 파일-씬 매핑 수: {len(target_file_scene_pairs)}, scenes: {target_scenes}")

    # Initialize NuScenes
    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False)
          

    
    config = config_factory('detection_cvpr_2019')
    
    # Load GT and Compare files
    print(f'🔍 Loading GT Boxes from: {args.gt}')
    gt_boxes, _ = load_gt(nusc, eval_set_map[args.version], DetectionBox, verbose=args.verbose)

    print(f'🔍 Loading Initial Boxes from: {args.ctrl}')
    ctrl_boxes, _ = load_prediction(args.ctrl, 
                                    config.max_boxes_per_sample, 
                                    DetectionBox,
                                    verbose=args.verbose)

    print(f'🔍 Loading Target Boxes from: {args.tar}')
    tar_boxes = extract_boxes_from_json_to_evalboxes_multiple(nusc, target_file_scene_pairs)
    print(f"✅ target files에서 {len(tar_boxes.sample_tokens)}개 샘플, {len(tar_boxes.all)}개 박스 추출 완료")
    
    # Filter to common scenes
    print("🔍 Filtering to common scenes...")
    gt_boxes, tar_boxes = filter_boxes_by_common_scenes(nusc, gt_boxes, tar_boxes)
    ctrl_boxes, tar_boxes = filter_boxes_by_common_scenes(nusc, ctrl_boxes, tar_boxes)

    # Prepare results storage
    results = []
           
    # Extract boxes from JSON file

    if len(tar_boxes.sample_tokens) == 0:
        print(f"❌ No boxes extracted from {args.tar}")
        return
    
    # Perform correspondence matching
    if args.verbose:
        print("🔄 Performing correspondence matching...")            
            
    # Perform evaluation
    if args.verbose:
        print(f"\n    Performing target evaluation...")


    eval_results = perform_evaluation(gt_boxes, tar_boxes, tar_boxes.sample_tokens, config, nusc, args.tar)
    if args.verbose:
        print(f"\n    Performing control evaluation...")

    output_dir = Path(args.ctrl).parents[0]
    ctrl_boxes_matched = match_boxes(tar_boxes, ctrl_boxes)
    ctrl_eval_results = perform_evaluation(gt_boxes, ctrl_boxes_matched, ctrl_boxes_matched.sample_tokens, config, nusc, output_dir)

    # Extract iteration number from filename and parse metrics
    match = re.search(r'(\d+)', args.name)
    iteration_number = int(match.group(1)) if match else 80000

    result_entry = {
        'file_path': args.tar,
        'file_name': os.path.basename(args.tar),
        'directory': args.tar,
        **eval_results,
    }
    results.append(result_entry)
    
    if args.verbose:
        print(f"✅ Evaluation completed for {args.tar}")
        print(f"   ATE: {eval_results['ATE']:.4f}, AOE: {eval_results['AOE']:.4f}, ASE: {eval_results['ASE']:.4f}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Set output path
        if args.output is None:
            output_path = os.path.join(args.tar, f"instance_wise_eval_{args.name.replace('.', '_')}.csv")
        else:
            output_path = args.output
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
    else:
        print("❌ No results to save")


if __name__ == "__main__":
    main()
