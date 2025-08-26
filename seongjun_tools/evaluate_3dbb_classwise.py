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
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics, DetectionMetricData
from nuscenes.eval.detection.algo import calc_ap
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean



# Import functions from other modules
import sys
sys.path.append('/workspace/drivestudio')
from seongjun_tools.generate_selected_instances import correspondence, filter_boxes_by_common_scenes
from seongjun_tools.evaluate_3dbb import filter_eval_boxes, load_gt, filter_boxes_by_scene
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

def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> Tuple[DetectionMetricData, Dict[str, List[float]]]:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions(), {}

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    match_pred_boxes = defaultdict(list)

    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            if match_gt_idx is not None:
                gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

                match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
                match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
                match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

                match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
                match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions(), {}

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    match_data_copy = copy.deepcopy(match_data)
    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err']), match_data_copy

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
                      config, nusc: NuScenes, target_path: str) -> Dict[str, float]:
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
    
    match_data_list = {}
    fields = ('trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err', 'conf')
    node_match_data_list = {
        name: {f: [] for f in fields}
        for name in ('RigidNodes', 'DeformableNodes', 'SMPLNodes')
    }    
    key_name_mapping = {ModelType.RigidNodes: 'RigidNodes', ModelType.DeformableNodes: 'DeformableNodes', ModelType.SMPLNodes: 'SMPLNodes'}
    for class_name in config.class_names:
        md, match_data_copy = accumulate(gt_boxes, tar_boxes, class_name, config.dist_fcn_callable, dist_th)
        metric_data_list.set(class_name, dist_th, md)
        match_data_list[class_name] = match_data_copy

        inv_key = detection_mapping_inv.get(class_name)
        if inv_key is not None and inv_key in OBJECT_CLASS_NODE_MAPPING:
            node_type = OBJECT_CLASS_NODE_MAPPING[inv_key]
            if node_type in (ModelType.RigidNodes, ModelType.DeformableNodes, ModelType.SMPLNodes):
                bucket = node_match_data_list[key_name_mapping[node_type]]
                for f in fields:
                    if f in match_data_copy:
                        bucket[f].extend(match_data_copy[f])

    # Calculate metrics
    metrics = DetectionMetrics(config)
    for class_name in config.class_names:
        # Compute APs.
        metric_data = metric_data_list[(class_name, dist_th)]
        ap = calc_ap(metric_data, config.min_recall, config.min_precision)
        metrics.add_label_ap(class_name, dist_th, ap)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            elif metric_name not in match_data_list[class_name]:
                tp = np.nan
            else:
                # metric_data = metric_data_list[(class_name, config.dist_th_tp)]
                # tp = calc_tp(metric_data, config.min_recall, metric_name)
                tp = float(np.mean(match_data_list[class_name][metric_name]))
            metrics.add_label_tp(class_name, metric_name, tp)
    
    for node_type, match_data in node_match_data_list.items():
        for metric_name in TP_METRICS:
            if metric_name not in match_data:
                tp = np.nan
            elif metric_name not in match_data:
                tp = np.nan
            else:
                tp = float(np.mean(match_data[metric_name]))
                std_dev = float(np.std(match_data[metric_name]))
                rmse = float(np.sqrt(np.mean(np.square(match_data[metric_name]))))
                print(f"🔍 {node_type} {metric_name} MAE: {tp}, RMSE: {rmse}")
            metrics.add_label_tp(node_type, metric_name, tp)

    # Get metrics summary
    metrics_summary = metrics.serialize()

    # Dump the metric data, meta and metrics to disk.
    output_dir = Path(target_path).parents[0]
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(output_dir, 'metrics_details.json'), 'w') as f:
        json.dump(metric_data_list.serialize(), f, indent=2)

    num_matched_boxes = 0
    for match_data in match_data_list.values():
        if 'trans_err' in match_data:
            num_matched_boxes += len(match_data['trans_err'])

    return {
        'ATE': metrics_summary['tp_errors']['trans_err'],
        'AOE': metrics_summary['tp_errors']['orient_err'],
        'ASE': metrics_summary['tp_errors']['scale_err'],
        'mAP': metrics_summary['mean_ap'],
        'NDS': metrics_summary['nd_score'],
        'AVE': metrics_summary['tp_errors']['vel_err'],
        'AAE': metrics_summary['tp_errors']['attr_err'],
        'num_tar_boxes': len(tar_boxes.all),
        'num_gt_boxes': len(gt_boxes.all),
        'num_matched_boxes': num_matched_boxes
    }

def parse_metrics_json(metrics_path: str, target_iteration: int) -> Dict[str, Optional[float]]:
    """metrics.json 파일에서 특정 iteration의 psnr과 ssim 값을 파싱합니다.
    
    Args:
        metrics_path: metrics.json 파일 경로
        target_iteration: 찾을 iteration 번호
        
    Returns:
        psnr과 ssim 값을 포함한 딕셔너리
    """
    result: Dict[str, Optional[float]] = {'psnr': None, 'ssim': None}
    
    if not os.path.exists(metrics_path):
        print(f"❌ metrics.json 파일을 찾을 수 없습니다: {metrics_path}")
        return result
    
    try:
        with open(metrics_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if data.get('iteration') == target_iteration:
                        result['psnr'] = data.get('train_metrics/psnr', None)
                        result['ssim'] = data.get('losses/ssim_loss', None)
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ metrics.json 파일 읽기 중 오류 발생: {e}")
    
    return result

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
        # default='/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred_class.json',
        # default='/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise/mini/001/instances/instances_info_pred.json',
        default='/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini/001/instances/instances_info_pred.json',
        # default='/workspace/drivestudio/output/box_experiments_0804_eval/prediction/results_tracking.json',
        help="Path to comparison prediction json file",
    )
    parser.add_argument(
        "--tar",
        type=str,
        # default='/workspace/drivestudio/output/box_experiments_0804_eval',
        # default='/workspace/drivestudio/output/box_experiments_0813',
        # default='/workspace/drivestudio/output/box_experiments_0821',
        default='/workspace/drivestudio/output/box_experiments_0825',
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
            print(f"\n    Performing target evaluation...")
        eval_results = perform_evaluation(final_gt_evalboxes, final_target_evalboxes, config, nusc, target_file)
        if args.verbose:
            print(f"\n    Performing control evaluation...")
        ctrl_eval_results = perform_evaluation(final_gt_evalboxes, final_ctrl_evalboxes, config, nusc, args.ctrl)

        # Store results
        # Get parent directory name (1st level up from file)
        path_parts = Path(target_file).parts
        try: 
            directory = path_parts[-3]
            metrics_path = Path(target_file).parents[1] / 'metrics.json'
        except:
            directory = Path(target_file).parents[1]
            metrics_path = Path(target_file).parents[1] / 'metrics.json'

        # Extract iteration number from filename and parse metrics
        match = re.search(r'(\d+)', args.name)
        iteration_number = int(match.group(1)) if match else 80000

        psnr, ssim = None, None
        if iteration_number is not None:
            if os.path.exists(metrics_path):
                metrics_data = parse_metrics_json(str(metrics_path), iteration_number)
                psnr, ssim = metrics_data['psnr'], metrics_data['ssim']
                if args.verbose:
                    print(f"📊 Iteration {iteration_number}: PSNR={psnr}, SSIM={ssim}")
            else:
                if args.verbose:
                    print(f"❌ metrics.json 파일을 찾을 수 없습니다: {metrics_path}")
        else:
            if args.verbose:
                print(f"❌ 파일명에서 iteration 번호를 추출할 수 없습니다: {args.name}")

        result_entry = {
            'file_path': target_file,
            'file_name': os.path.basename(target_file),
            'directory': directory,
            **eval_results,
            'psnr': psnr,
            'ssim': ssim
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
