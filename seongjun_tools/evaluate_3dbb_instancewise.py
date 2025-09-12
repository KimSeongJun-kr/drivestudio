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
from datasets.base.scene_dataset import ModelType

# from datasets.nuscenes.nuscenes_sourceloader import OBJECT_CLASS_NODE_MAPPING
OBJECT_CLASS_NODE_MAPPING = {
    # Rigid objects (vehicles)
    "vehicle.bus.bendy": ModelType.RigidNodes,
    "vehicle.bus.rigid": ModelType.RigidNodes,
    "vehicle.car": ModelType.RigidNodes,
    "vehicle.construction": ModelType.RigidNodes,
    "vehicle.emergency.ambulance": ModelType.RigidNodes,
    "vehicle.emergency.police": ModelType.RigidNodes,
    "vehicle.motorcycle": ModelType.RigidNodes,
    "vehicle.trailer": ModelType.RigidNodes,
    "vehicle.truck": ModelType.RigidNodes,

    # Humans (SMPL model)
    "human.pedestrian.adult": ModelType.SMPLNodes,
    "human.pedestrian.child": ModelType.SMPLNodes,
    "human.pedestrian.construction_worker": ModelType.SMPLNodes,
    "human.pedestrian.police_officer": ModelType.SMPLNodes,

    # Potentially deformable objects
    "human.pedestrian.personal_mobility": ModelType.DeformableNodes,
    "human.pedestrian.stroller": ModelType.DeformableNodes,
    "human.pedestrian.wheelchair": ModelType.DeformableNodes,
    "animal": ModelType.DeformableNodes,
    "vehicle.bicycle": ModelType.DeformableNodes
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

def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               verbose: bool = False) -> Tuple[DetectionMetricData, Dict[str, List[float]]]:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, gt_boxes.all[0].detection_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions(), {}

    # Organize the predictions in a single list.
    pred_dists = [box.ego_dist for box in pred_boxes.all]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_dists), pred_boxes.all[0].detection_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    assert gt_boxes.all[0].instance_token == pred_boxes.all[0].instance_token, "GT and Pred boxes must have the same instance token"

    # Sort by distance
    pred_boxes_list = pred_boxes.all

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    dist = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'dist': [],
                  'sample_tokens': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    match_pred_boxes = defaultdict(list)

    for ind in range(len(pred_boxes_list)):
        pred_box = pred_boxes_list[ind]
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box.instance_token == pred_box.instance_token and not (pred_box.sample_token, gt_idx) in taken:
                match_gt_idx = gt_idx
                break

        if match_gt_idx is not None:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            dist.append(pred_box.ego_dist)

            # Since it is a match, update match data also.
            if match_gt_idx is not None:
                gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

                match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
                match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
                match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if pred_box.detection_name == 'barrier' else 2 * np.pi
                match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

                match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
                match_data['dist'].append(pred_box.ego_dist)
                match_data['sample_tokens'].append(pred_box.sample_token)
        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            dist.append(pred_box.ego_dist)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions(), {}

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_dists))][::-1]
    dist_sorted_tp = [tp[i] for i in sortind]
    dist_sorted_fp = [fp[i] for i in sortind]
    dist_sorted_dist = [dist[i] for i in sortind]


    # Accumulate.   
    dist_sorted_tp = np.cumsum(dist_sorted_tp).astype(float)
    dist_sorted_fp = np.cumsum(dist_sorted_fp).astype(float)
    dist_sorted_dist = np.array(dist_sorted_dist)

    # Calculate precision and recall.
    prec = dist_sorted_tp / (dist_sorted_fp + dist_sorted_tp)
    rec = dist_sorted_tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    dist = np.interp(rec_interp, rec, dist_sorted_dist, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    match_data_copy = copy.deepcopy(match_data)
    for key in match_data.keys():
        if key == "dist" or key == "sample_tokens":
            continue  # Distance is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(dist[::-1], match_data['dist'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=dist,
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
                    detection_name = box.get('detection_name', '')
                    if detection_name == 'human.pedestrian.personal_mobility' or detection_name == '':
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


def perform_evaluation(gt_boxes: EvalBoxes, tar_boxes: EvalBoxes, sample_tokens: List[str],
                      config, nusc: NuScenes, output_dir: str = '') -> Dict[str, float]:
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
    tar_boxes_filtered = filter_eval_boxes(nusc, tar_boxes, 50, verbose=False)
    gt_boxes_filtered = filter_eval_boxes(nusc, gt_boxes, 50, verbose=False)

    instance_wise_tar_boxes = create_instances_wise_dicts(tar_boxes_filtered, sample_tokens)
    instance_wise_gt_boxes = create_instances_wise_dicts(gt_boxes_filtered, sample_tokens)
    
    # Accumulate metric data
    instance_wise_metric_data_list = DetectionMetricDataList()
    dist_th = 1.0

    class_wise_err_data_list: Dict[str, Dict[str, List[float]]] = {}    
    instance_wise_err_data_list: Dict[str, Tuple[str, Dict[str, List[float]], EvalBoxes, EvalBoxes]] = {}
    fields = ('trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err', 'conf')
    nodeclass_wise_err_data_list = {
        name: {f: [] for f in fields}
        for name in ('RigidNodes', 'DeformableNodes', 'SMPLNodes')
    }    
    key_name_mapping = {ModelType.RigidNodes: 'RigidNodes', ModelType.DeformableNodes: 'DeformableNodes', ModelType.SMPLNodes: 'SMPLNodes'}

    for instance_token in instance_wise_tar_boxes.keys():
        tar_boxes_instance = instance_wise_tar_boxes[instance_token]
        if instance_token not in instance_wise_gt_boxes:
            continue
        gt_boxes_instance = instance_wise_gt_boxes[instance_token]
        metric_data, raw_err_data = accumulate(gt_boxes_instance, tar_boxes_instance)

        if tar_boxes_instance.all[0].detection_name not in class_wise_err_data_list:
            class_wise_err_data_list[tar_boxes_instance.all[0].detection_name] = {'trans_err': [],
                                                                          'vel_err': [],
                                                                          'scale_err': [],
                                                                          'orient_err': [],
                                                                          'attr_err': [],
                                                                          'conf': []}
        for f in fields:
            if f in raw_err_data:
                class_wise_err_data_list[tar_boxes_instance.all[0].detection_name][f].extend(raw_err_data[f])

        instance_idx = tar_boxes_instance.all[0].instance_idx
        instance_wise_metric_data_list.set(str(instance_idx), dist_th, metric_data)
        instance_wise_err_data_list[instance_token] = (instance_idx, raw_err_data, gt_boxes_instance, tar_boxes_instance)

        class_name = tar_boxes_instance.all[0].detection_name
        inv_key = detection_mapping_inv.get(class_name)
        if inv_key is not None and inv_key in OBJECT_CLASS_NODE_MAPPING:
            node_type = OBJECT_CLASS_NODE_MAPPING[inv_key]
            if node_type in (ModelType.RigidNodes, ModelType.DeformableNodes, ModelType.SMPLNodes):
                bucket = nodeclass_wise_err_data_list[key_name_mapping[node_type]]
                for f in fields:
                    if f in raw_err_data:
                        bucket[f].extend(raw_err_data[f])

    # Calculate metrics
    metrics_class_wise = DetectionMetrics(config)
    metrics_instance_wise = DetectionMetrics(config)

    all_tp_errors = {metric_name: [] for metric_name in TP_METRICS}
    for node_type, raw_err_data in nodeclass_wise_err_data_list.items():
        # ap = calc_ap(raw_err_data, config.min_recall, config.min_precision)
        # metrics_class_wise.add_label_ap(node_type, dist_th, ap)

        for metric_name in TP_METRICS:
            if metric_name not in raw_err_data:
                tp = np.nan
                rmse = np.nan
            else:
                tp = float(np.mean(raw_err_data[metric_name]))
                rmse = float(np.sqrt(np.mean(np.square(raw_err_data[metric_name]))))
                all_tp_errors[metric_name].extend(raw_err_data[metric_name])
                # print(f"🔍 {node_type} {metric_name} MAE: {tp}, rmse: {rmse}")
            metrics_class_wise.add_label_tp(node_type, metric_name, tp)
            metrics_class_wise.add_label_tp_RMSE(node_type, metric_name, rmse)
    for metric_name in TP_METRICS:
        all_tp = float(np.mean(all_tp_errors[metric_name]))
        all_rmse = float(np.sqrt(np.mean(np.square(all_tp_errors[metric_name]))))
        metrics_class_wise.add_label_tp('all', metric_name, all_tp)
        metrics_class_wise.add_label_tp_RMSE('all', metric_name, all_rmse)

    # for class_name in config.class_names:
    #     # Compute APs.
    #     metric_data = instance_wise_metric_data_list[(class_name, dist_th)]
    #     ap = calc_ap(metric_data, config.min_recall, config.min_precision)
    #     metrics.add_label_ap(class_name, dist_th, ap)

    #     # Compute TP metrics.
    #     for metric_name in TP_METRICS:
    #         if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
    #             tp = np.nan
    #             rmse = np.nan
    #         elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
    #             tp = np.nan
    #             rmse = np.nan
    #         elif metric_name not in class_wise_err_data_list[class_name]:
    #             tp = np.nan
    #             rmse = np.nan
    #         else:
    #             # metric_data = metric_data_list[(class_name, config.dist_th_tp)]
    #             # tp = calc_tp(metric_data, config.min_recall, metric_name)
    #             tp = float(np.mean(class_wise_err_data_list[class_name][metric_name]))
    #             rmse = float(np.sqrt(np.mean(np.square(class_wise_err_data_list[class_name][metric_name]))))
    #         metrics.add_label_tp(class_name, metric_name, tp)
    #         metrics.add_label_tp_RMSE(class_name, metric_name, rmse)

    all_tp_errors = {metric_name: [] for metric_name in TP_METRICS}
    for instance_token, (instance_idx, raw_err_data, gt_boxes_instance, tar_boxes_instance) in instance_wise_err_data_list.items():
        for metric_name in TP_METRICS:
            if metric_name not in raw_err_data:
                tp = np.nan
                rmse = np.nan
            elif metric_name not in raw_err_data:
                tp = np.nan
                rmse = np.nan
            else:
                tp = float(np.mean(raw_err_data[metric_name]))
                rmse = float(np.sqrt(np.mean(np.square(raw_err_data[metric_name]))))
                # print(f"🔍 {instance_idx} {metric_name} MAE: {tp}, std_dev: {std_dev}")
                all_tp_errors[metric_name].extend(raw_err_data[metric_name])
            metrics_instance_wise.add_label_tp(str(instance_token), metric_name, tp)
            metrics_instance_wise.add_label_tp_RMSE(str(instance_token), metric_name, rmse)
    for metric_name in TP_METRICS:
        all_tp = float(np.mean(all_tp_errors[metric_name]))
        all_rmse = float(np.sqrt(np.mean(np.square(all_tp_errors[metric_name]))))
        metrics_instance_wise.add_label_tp('all', metric_name, all_tp)
        metrics_instance_wise.add_label_tp_RMSE('all', metric_name, all_rmse)

    # Get metrics summary
    metrics_summary_class_wise = metrics_class_wise.serialize()
    metrics_summary_instance_wise = metrics_instance_wise.serialize()

    if output_dir != '':
        # Dump the metric data, meta and metrics to disk.
        with open(os.path.join(output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(instance_wise_metric_data_list.serialize(), f, indent=2)
        with open(os.path.join(output_dir, 'metrics_summmary_class_wise.json'), 'w') as f:
            json.dump(metrics_summary_class_wise, f, indent=2)
        with open(os.path.join(output_dir, 'metrics_summmary_instance_wise.json'), 'w') as f:
            json.dump(metrics_summary_instance_wise, f, indent=2)


        instance_wise_frame_err_data = {}
        for instance_token, (instance_idx, raw_err_data, gt_boxes_instance, tar_boxes_instance) in instance_wise_err_data_list.items():
            frames = []
            for sample_token in raw_err_data['sample_tokens']:
                sample_idx = sample_tokens.index(sample_token)
                frames.append(sample_idx * 5)

            instance_wise_frame_err_data[instance_token] = {
                'frames': frames,
                'trans_err': raw_err_data['trans_err'],
                'vel_err': raw_err_data['vel_err'],
                'scale_err': raw_err_data['scale_err'],
                'orient_err': raw_err_data['orient_err'],
                'attr_err': raw_err_data['attr_err'],
                'dist': raw_err_data['dist'],
                'instance_token': instance_token,
                'instance_idx': int(instance_idx),
                'num_gaussians': tar_boxes_instance.all[0].num_gaussians,
                'detection_name': tar_boxes_instance.all[0].detection_name,
            }
        with open(os.path.join(output_dir, 'instance_wise_frame_err_data.json'), 'w') as f:
            json.dump(instance_wise_frame_err_data, f, indent=2)
            
        print(f"✅ results saved to {output_dir}")


    num_matched_boxes = 0
    for (instance_idx, raw_err_data, gt_boxes_instance, tar_boxes_instance) in instance_wise_err_data_list.values():
        if 'trans_err' in raw_err_data:
            num_matched_boxes += len(raw_err_data['trans_err'])

    return {
        'ATE': metrics_summary_class_wise['label_tp_MAE_errors']['all']['trans_err'],
        'AOE': metrics_summary_class_wise['label_tp_MAE_errors']['all']['orient_err'],
        'ASE': metrics_summary_class_wise['label_tp_MAE_errors']['all']['scale_err'],
        # 'mAP': metrics_summary_class_wise['mean_ap'],
        # 'NDS': metrics_summary_class_wise['nd_score'],
        # 'AVE': metrics_summary_class_wise['tp_errors']['vel_err'],
        # 'AAE': metrics_summary_class_wise['tp_errors']['attr_err'],
        'ATE_RMSE': metrics_summary_class_wise['label_tp_RMSE_errors']['all']['trans_err'],
        'AOE_RMSE': metrics_summary_class_wise['label_tp_RMSE_errors']['all']['orient_err'],
        'ASE_RMSE': metrics_summary_class_wise['label_tp_RMSE_errors']['all']['scale_err'],
        # 'AVE_RMSE': metrics_summary_class_wise['tp_RMSE_errors']['vel_err'],
        # 'AAE_RMSE': metrics_summary_class_wise['tp_RMSE_errors']['attr_err'],
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

def match_boxes(tar_boxes: EvalBoxes, src_boxes: EvalBoxes) -> EvalBoxes:
    """target_boxes와 일치하는 sample_token과 instance_token을 가진 ctrl_boxes만 골라냅니다.
    
    Args:
        tar_boxes: 기준이 되는 박스들
        src_boxes: 매칭할 박스들
        
    Returns:
        tar_boxes와 매칭되는 src_boxes 포함한 EvalBoxes 객체
    """
    matched_boxes = EvalBoxes()
    
    # target_boxes의 각 sample_token과 instance_token 조합을 수집
    target_combinations = set()
    for sample_token in tar_boxes.sample_tokens:
        for box in tar_boxes[sample_token]:
            target_combinations.add((sample_token, box.instance_token))
    
    # ctrl_boxes에서 매칭되는 박스들만 선택
    for sample_token in src_boxes.sample_tokens:
        matched_boxes_for_sample = []
        for box in src_boxes[sample_token]:
            if (sample_token, box.instance_token) in target_combinations:
                matched_boxes_for_sample.append(box)
        
        # 매칭된 박스가 있는 경우에만 추가
        if matched_boxes_for_sample:
            matched_boxes.add_boxes(sample_token, matched_boxes_for_sample)
    
    return matched_boxes

def create_instances_wise_dicts(boxes: EvalBoxes, sample_tokens: List[str]) -> Dict[str, EvalBoxes]:
    instances_boxes: Dict[str, EvalBoxes] = {}
    for sample_token in sample_tokens:
        if sample_token not in boxes.sample_tokens:
            continue
        for box in boxes[sample_token]:
            if box.instance_token not in instances_boxes:
                instances_boxes[box.instance_token] = EvalBoxes()
            instances_boxes[box.instance_token].add_boxes(sample_token, [box])
    return instances_boxes

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
        default='/workspace/drivestudio/output/box_experiments_0826',
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
    scene_sample_tokens = _get_scene_sample_tokens_chronologically(nusc, args.scene_name)

    eval_set_map = {
        'v1.0-mini': 'mini_trainval',
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
        
        # Extract boxes from JSON file
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
               
        # Perform evaluation
        if args.verbose:
            print(f"\n    Performing target evaluation...")
        output_dir = Path(target_file).parents[0]
        eval_results = perform_evaluation(gt_boxes, target_boxes, scene_sample_tokens, config, nusc, output_dir)
        if args.verbose:
            print(f"\n    Performing control evaluation...")

        output_dir = Path(args.ctrl).parents[0]
        ctrl_boxes_matched = match_boxes(target_boxes, ctrl_boxes)
        ctrl_eval_results = perform_evaluation(gt_boxes, ctrl_boxes_matched, scene_sample_tokens, config, nusc, output_dir)

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
            output_path = os.path.join(args.tar, f"instance_wise_eval_{args.name.replace('.', '_')}.csv")
        else:
            output_path = args.output
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
    else:
        print("❌ No results to save")


if __name__ == "__main__":
    main()
