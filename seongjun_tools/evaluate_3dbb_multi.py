import argparse
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

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.detection.algo import calc_ap
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist

# Import functions from other modules
import sys
sys.path.append('/workspace/drivestudio')
from seongjun_tools.generate_selected_instances import correspondence, filter_boxes_by_common_scenes
from seongjun_tools.evaluate_3dbb import accumulate, filter_eval_boxes, load_gt, filter_boxes_by_scene

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

def extract_all_boxes_from_json(json_path: str) -> Optional[Dict[int, List]]:
    """box_poses_*.json 파일에서 모든 프레임의 박스 정보를 추출합니다."""

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

    return frame_boxes

def extract_boxes_from_json_to_evalboxes(json_path: str, sample_tokens: List[str]) -> EvalBoxes:
    """JSON 파일에서 박스 정보를 추출하여 EvalBoxes 형태로 변환합니다.
    
    Args:
        json_path: JSON 파일 경로
        sample_tokens: 사용할 sample token 리스트
        
    Returns:
        EvalBoxes 객체
    """
    frame_boxes = extract_all_boxes_from_json(json_path)
    if frame_boxes is None:
        print(f"❌ JSON 파일에서 박스 정보를 추출할 수 없습니다: {json_path}")
        return EvalBoxes()
    
    eval_boxes = EvalBoxes()
    
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
                    
                    # box 정보를 DetectionBox로 변환
                    detection_box = DetectionBox(
                        sample_token=sample_token,
                        translation=box.get('translation', [0, 0, 0]),
                        size=box.get('size', [1, 1, 1]),
                        rotation=box.get('rotation', [1, 0, 0, 0]),
                        velocity=box.get('velocity', [0, 0]),
                        detection_name=box.get('detection_name', 'car'),
                        detection_score=box.get('detection_score', 0.5),
                        attribute_name=box.get('attribute_name', '')
                    )
                    detection_boxes.append(detection_box)
                except Exception as e:
                    print(f"❌ 박스 변환 중 오류 발생: {e}")
                    print(f"   박스 정보: {box}")
                    continue
            
            if detection_boxes:
                eval_boxes.add_boxes(sample_token, detection_boxes)
    
    return eval_boxes


def find_files_with_name(directory: str, filename: str) -> List[str]:
    """지정된 디렉토리의 하위 경로에서 특정 이름의 파일들을 찾습니다.
    
    Args:
        directory: 검색할 디렉토리
        filename: 찾을 파일 이름
        
    Returns:
        파일 경로 리스트
    """
    pattern = os.path.join(directory, "**", filename)
    files = glob.glob(pattern, recursive=True)
    
    # 경로를 정렬
    files.sort()
    
    print(f"📁 '{filename}' 파일을 찾았습니다 ({len(files)}개):")
    for file_path in files:
        print(f"  - {file_path}")
    
    return files


def perform_evaluation(gt_boxes: EvalBoxes, tar_boxes: EvalBoxes,
                      config, nusc: NuScenes) -> Dict[str, float]:
    """3D 바운딩 박스 평가를 수행합니다.
    
    Args:
        gt_boxes: Ground truth boxes
        pred_boxes: Prediction boxes
        sample_ann_tokens: Sample annotation tokens
        config: 평가 설정
        nusc: NuScenes 객체
        
    Returns:
        평가 결과 딕셔너리
    """
    # Add center distances
    tar_boxes = add_center_dist(nusc, tar_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    
    # Filter boxes (distance, points per box, etc.)
    tar_boxes = filter_eval_boxes(nusc, tar_boxes, 50, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, 50, verbose=False)
    
    # Accumulate metric data
    metric_data_list = DetectionMetricDataList()
    dist_th = 1.0
    
    md, match_data_copy = accumulate(
        gt_boxes, tar_boxes, config.dist_fcn_callable, dist_th
    )
    
    metric_data_list.set('all', dist_th, md)
    
    # Calculate metrics
    metrics = DetectionMetrics(config)
    
    # Compute AP
    metric_data = metric_data_list[('all', dist_th)]
    ap = calc_ap(metric_data, config.min_recall, config.min_precision)
    metrics.add_label_ap('all', dist_th, ap)
    
    # Compute TP metrics
    tp_metrics = {}
    for metric_name in TP_METRICS:
        tp = float(np.mean(match_data_copy[metric_name]))
        metrics.add_label_tp('all', metric_name, tp)
        tp_metrics[metric_name] = tp
    
    # Get metrics summary
    metrics_summary = metrics.serialize()
    
    return {
        'ATE': tp_metrics['trans_err'],
        'AOE': tp_metrics['orient_err'],
        'ASE': tp_metrics['scale_err'],
        'mAP': metrics_summary['mean_ap'],
        'NDS': metrics_summary['nd_score'],
        'AVE': tp_metrics['vel_err'],
        'AAE': tp_metrics['attr_err'],
        'num_tar_boxes': len(tar_boxes.all),
        'num_gt_boxes': len(gt_boxes.all),
        'num_matched_boxes': len(match_data_copy['trans_err'])
    }

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
        default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred.json',
        help="Path to comparison prediction json file",
    )
    parser.add_argument(
        "--tar",
        type=str,
        default='/workspace/drivestudio/output/box_experiments_0702',
        help="Directory to search for target files",
    )
    parser.add_argument(
        "--name",
        type=str,
        default='box_poses_80000.json',
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
        default=False,
        help="Verbose",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        # default=None,
        default='scene-0103',
        help="Scene name to filter boxes (e.g., 'scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', "
                                                "'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', "
                                                "'scene-1094', 'scene-1100')",
    )

    args = parser.parse_args()

    # Initialize NuScenes
    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False)
    
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
    
    # Verify input files exist
    if args.gt is not None:
        assert os.path.exists(args.gt), f'Error: GT file does not exist: {args.gt}'
    if args.ctrl is not None:
        assert os.path.exists(args.ctrl), f'Error: Compare file does not exist: {args.ctrl}'
    if args.tar is not None:
        assert os.path.exists(args.tar), f'Error: Tar directory does not exist: {args.tar}'
    
    config = config_factory('detection_cvpr_2019')
    
    # Load GT and Compare files
    print(f'🔍 Loading GT prediction from: {args.gt}')
    # gt_boxes, _ = load_prediction(args.gt, 
    #                             config.max_boxes_per_sample, 
    #                             DetectionBox,
    #                             verbose=args.verbose)
    gt_boxes, _ = load_gt(nusc, eval_set_map[args.version], DetectionBox, verbose=args.verbose)

    
    print(f'🔍 Loading Compare prediction from: {args.ctrl}')
    ctrl_boxes, _ = load_prediction(args.ctrl, 
                                    config.max_boxes_per_sample, 
                                    DetectionBox,
                                    verbose=args.verbose)
    
    # Filter by scene if specified
    if args.scene_name:
        print(f"🔍 Filtering boxes by scene: {args.scene_name}")
        gt_boxes = filter_boxes_by_scene(nusc, gt_boxes, args.scene_name)
        ctrl_boxes = filter_boxes_by_scene(nusc, ctrl_boxes, args.scene_name)
    else:
        # Filter to common scenes
        print("🔍 Filtering to common scenes...")
        gt_boxes, ctrl_boxes = filter_boxes_by_common_scenes(nusc, gt_boxes, ctrl_boxes)
    
    # Match GT (src) with Control (tar1)
    dist_th = 1.0
    matched_gt_w_ctrl, matched_strl_w_gt = correspondence(
        ctrl_boxes, gt_boxes, config.dist_fcn_callable, dist_th)
    # Create mappings
    gt_to_ctrl_mapping = defaultdict(dict)
    for sample_token in matched_gt_w_ctrl.keys():
        gt_boxes_list = matched_gt_w_ctrl[sample_token]
        ctrl_boxes_list = matched_strl_w_gt[sample_token]
        for gt_box, ctrl_box in zip(gt_boxes_list, ctrl_boxes_list):
            gt_key = (gt_box.translation[0], gt_box.translation[1], gt_box.translation[2],
                        gt_box.size[0], gt_box.size[1], gt_box.size[2])
            gt_to_ctrl_mapping[sample_token][gt_key] = ctrl_box

    # Find all target files
    print(f'🔍 Finding {args.name} files in {args.tar}...')
    target_files = find_files_with_name(args.tar, args.name)
    
    if not target_files:
        print(f"❌ No {args.name} files found in {args.tar}")
        return
    
    # Prepare results storage
    results = []
    
    # Process each target file
    for target_file in tqdm.tqdm(target_files, desc="Processing target files"):
        if args.verbose:
            print(f"\n📊 Processing: {target_file}")
        
        # Get common sample tokens from gt and compare
        common_samples_gt_ctrl = list(set(gt_boxes.sample_tokens).intersection(set(ctrl_boxes.sample_tokens)))
        
        if not common_samples_gt_ctrl:
            print(f"❌ No common sample tokens found for {target_file}")
            continue
        
        # Extract boxes from JSON file
        scene_sample_tokens = _get_scene_sample_tokens_chronologically(nusc, args.scene_name)
        target_boxes = extract_boxes_from_json_to_evalboxes(target_file, scene_sample_tokens)
        if args.verbose:
            print(f"✅ JSON에서 {len(target_boxes.sample_tokens)}개 샘플, {len(target_boxes.all)}개 박스 추출 완료")
        if target_boxes is not None:
            target_boxes = filter_boxes_by_scene(nusc, target_boxes, args.scene_name)
        
        if len(target_boxes.sample_tokens) == 0:
            print(f"❌ No boxes extracted from {target_file}")
            continue
        
        # Perform correspondence matching
        if args.verbose:
            print("🔄 Performing correspondence matching...")            
        # Match GT (src) with Target (tar2)
        matched_gt_w_tar, matched_tar_w_gt = correspondence(
            target_boxes, gt_boxes, config.dist_fcn_callable, dist_th)
        
        # Find boxes that match both compare and target
        final_gt_boxes = defaultdict(list)
        final_ctrl_boxes = defaultdict(list)
        final_target_boxes = defaultdict(list)
        
        gt_to_tar_mapping = defaultdict(dict)
        for sample_token in matched_gt_w_tar.keys():
            gt_boxes_list = matched_gt_w_tar[sample_token]
            tar_boxes_list = matched_tar_w_gt[sample_token]
            for gt_box, tar_box in zip(gt_boxes_list, tar_boxes_list):
                gt_key = (gt_box.translation[0], gt_box.translation[1], gt_box.translation[2],
                            gt_box.size[0], gt_box.size[1], gt_box.size[2])
                gt_to_tar_mapping[sample_token][gt_key] = tar_box
        
        # Find boxes that match both control and target
        for sample_token, gt_boxes_list in matched_gt_w_tar.items():
            for gt_box in gt_boxes_list:
                gt_key = (gt_box.translation[0], gt_box.translation[1], gt_box.translation[2],
                            gt_box.size[0], gt_box.size[1], gt_box.size[2])
                
                # Check if this GT box also matches compare
                if gt_key in gt_to_ctrl_mapping[sample_token]:
                    final_gt_boxes[sample_token].append(gt_box)
                    final_ctrl_boxes[sample_token].append(gt_to_ctrl_mapping[sample_token][gt_key])
                    final_target_boxes[sample_token].append(gt_to_tar_mapping[sample_token][gt_key])
        
        # Convert to EvalBoxes
        final_gt_evalboxes = EvalBoxes()
        final_ctrl_evalboxes = EvalBoxes()
        final_target_evalboxes = EvalBoxes()
        
        for sample_token, boxes in final_gt_boxes.items():
            final_gt_evalboxes.add_boxes(sample_token, boxes)
        for sample_token, boxes in final_ctrl_boxes.items():
            final_ctrl_evalboxes.add_boxes(sample_token, boxes)
        for sample_token, boxes in final_target_boxes.items():
            final_target_evalboxes.add_boxes(sample_token, boxes)
        
        # Perform evaluation
        if args.verbose:
            print("📊 Performing evaluation...")
        eval_results = perform_evaluation(final_gt_evalboxes, final_target_evalboxes, config, nusc)
        
        # Store results
        # Get parent directory name (1st level up from file)
        path_parts = Path(target_file).parts
        directory = ''
        try: 
            directory = path_parts[-3]
        except:
            directory = os.path.dirname(target_file)

        result_entry = {
            'file_path': target_file,
            'file_name': os.path.basename(target_file),
            'directory': directory,
            **eval_results
        }
        results.append(result_entry)
        
        if args.verbose:
            print(f"✅ Evaluation completed for {target_file}")
            print(f"   ATE: {eval_results['ATE']:.4f}, AOE: {eval_results['AOE']:.4f}, ASE: {eval_results['ASE']:.4f}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Set output path
        if args.output is None:
            output_path = os.path.join(args.tar, f"evaluation_results_{args.name.replace('.', '_')}.csv")
        else:
            output_path = args.output
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
    else:
        print("❌ No results to save")


if __name__ == "__main__":
    main()
