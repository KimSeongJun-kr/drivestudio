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
import sys
sys.path.append('/workspace/drivestudio')
from seongjun_tools.splits import create_splits_scenes
# from nuscenes.utils.splits import create_splits_scenes
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

def filter_boxes_by_common_scenes(nusc: NuScenes, pred_boxes: EvalBoxes, gt_boxes: EvalBoxes) -> Tuple[EvalBoxes, EvalBoxes]:
    """src와 tar에 모두 포함된 scene들만으로 boxes를 필터링합니다.

    Args:
        nusc: NuScenes 객체
        pred_boxes: prediction EvalBoxes
        gt_boxes: ground truth EvalBoxes

    Returns:
        필터링된 (src_boxes, tar_boxes) 튜플
    """
    # 각각에서 scene들 추출
    pred_scenes = set(get_scenes_from_boxes(nusc, pred_boxes))
    gt_scenes = set(get_scenes_from_boxes(nusc, gt_boxes))
    
    # 공통 scene들 찾기
    common_scenes = pred_scenes.intersection(gt_scenes)
    
    print(f"📊 Pred scenes: {len(pred_scenes)}, GT scenes: {len(gt_scenes)}, Common scenes: {len(common_scenes)}")
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
    
    filtered_pred_boxes = filter_by_scene_tokens(pred_boxes, common_scene_tokens)
    filtered_gt_boxes = filter_by_scene_tokens(gt_boxes, common_scene_tokens)
    
    print(f"✅ 필터링 완료: pred samples: {len(filtered_pred_boxes.sample_tokens)}, gt samples: {len(filtered_gt_boxes.sample_tokens)}")
    
    return filtered_pred_boxes, filtered_gt_boxes

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
                    # continue
                    detection_name = 'car'

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

def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: float,
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    # class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist]
        dist_filter += len(eval_boxes[sample_token])

        # # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        # eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        # point_filter += len(eval_boxes[sample_token])

        # # Perform bike-rack filtering.
        # sample_anns = nusc.get('sample', sample_token)['anns']
        # bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
        #                  nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        # bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        # filtered_boxes = []
        # for box in eval_boxes[sample_token]:
        #     if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
        #         in_a_bikerack = False
        #         for bikerack_box in bikerack_boxes:
        #             if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
        #                 in_a_bikerack = True
        #         if not in_a_bikerack:
        #             filtered_boxes.append(box)
        #     else:
        #         filtered_boxes.append(box)

        # eval_boxes.boxes[sample_token] = filtered_boxes
        # bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        # print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        # print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def accumulate(gt_boxes: EvalBoxes,
                pred_boxes: EvalBoxes,
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
    npos = len([1 for gt_box in gt_boxes.all])
    if verbose:
        print("Found {} GT of all classes out of {} total across {} samples.".
              format(npos, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions(), {}

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of all class out of {} total across {} samples.".
              format(len(pred_confs), len(pred_boxes.all), len(pred_boxes.sample_tokens)))

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
            if not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            # print(min_dist)
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
                period = np.pi if gt_box_match.detection_name == 'barrier' else 2 * np.pi
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

    return filtered_boxes
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--pred",
        type=str,
        # default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_gt_pred.json",
        default="/workspace/drivestudio/output/ceterpoint_pose/results_nusc_matched_pred_selected_tar1.json",
        # default="/workspace/drivestudio/output/feasibility_check/updated/poses_selected_tar2.json",
        help="Path to prediction json",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Scene name to evaluate boxes (e.g., 'scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', "
                                                "'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', "
                                                "'scene-1094', 'scene-1100')",
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


    args = parser.parse_args()

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=args.verbose)
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }

    assert os.path.exists(args.pred), 'Error: The result file does not exist!'
    config = config_factory('detection_cvpr_2019')

    print(f"Loading prediction from: {args.pred}")
    pred_boxes, meta = load_prediction(args.pred, 
                                        config.max_boxes_per_sample, 
                                        DetectionBox,
                                        verbose=args.verbose)

    gt_boxes, sample_ann_tokens = load_gt(nusc, eval_set_map[args.version], DetectionBox, verbose=args.verbose)

    # Filter boxes by scene if scene_name is provided
    if args.scene_name:
        print(f"Filtering boxes by scene: {args.scene_name}")
        pred_boxes = filter_boxes_by_scene(nusc, pred_boxes, args.scene_name)
        gt_boxes = filter_boxes_by_scene(nusc, gt_boxes, args.scene_name)
        print(f"✅ Scene '{args.scene_name}'에서 {len(pred_boxes.sample_tokens)}개의 샘플을 찾았습니다.")
    else:
        # Filter to only include scenes that exist in both pred and gt
        print("Filtering to common scenes in both pred and gt...")
        pred_boxes, gt_boxes = filter_boxes_by_common_scenes(nusc, pred_boxes, gt_boxes)

    # Note: We no longer require exact sample token matches since we're filtering by common scenes
    # but we can still check if there are any overlapping samples
    common_samples = set(pred_boxes.sample_tokens).intersection(set(gt_boxes.sample_tokens))
    print(f"📊 Common samples between pred and gt: {len(common_samples)}")
    print(f"🔄 Total pred samples: {len(pred_boxes.sample_tokens)}, Total gt samples: {len(gt_boxes.sample_tokens)}")

    
    # Add center distances.
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)

    # Filter boxes (distance, points per box, etc.).
    num_pred_boxes_before_filtering = len(pred_boxes.all)
    if args.verbose:
        print('Filtering predictions')
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, 50, verbose=args.verbose)
    if args.verbose:
        print('Filtering ground truth annotations')
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, 50, verbose=args.verbose)

    print(f"num pred boxes before filtering: {num_pred_boxes_before_filtering}, after filtering: {len(pred_boxes.all)}")

    sample_tokens = gt_boxes.sample_tokens
    
    # return

    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    print('Accumulating metric data...')
    metric_data_list = DetectionMetricDataList()
    dist_th = 1.0

    md, match_data_copy = accumulate(gt_boxes, pred_boxes, config.dist_fcn_callable, dist_th)
    metric_data_list.set('all', dist_th, md)


    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    print('Calculating metrics...')
    metrics = DetectionMetrics(config)

    # Compute APs.
    metric_data = metric_data_list[('all', dist_th)]
    ap = calc_ap(metric_data, config.min_recall, config.min_precision)
    metrics.add_label_ap('all', dist_th, ap)

    # Compute TP metrics.
    for metric_name in TP_METRICS:
        # metric_data = metric_data_list[('all', dist_th)]
        # tp = calc_tp(metric_data, config.min_recall, metric_name)
        tp = float(np.mean(match_data_copy[metric_name]))
        metrics.add_label_tp('all', metric_name, tp)

    # Dump the metric data, meta and metrics to disk.
    if args.verbose:
        print('Saving metrics to: %s' % str(Path(args.pred).parent))
    metrics_summary = metrics.serialize()
    metrics_summary['meta'] = meta.copy()
    with open(os.path.join(str(Path(args.pred).parent), 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(str(Path(args.pred).parent), 'metrics_details.json'), 'w') as f:
        json.dump(metric_data_list.serialize(), f, indent=2)

    # Print high-level metrics.
    print('mAP: %.4f' % (metrics_summary['mean_ap']))
    print('NDS: %.4f' % (metrics_summary['nd_score']))

    # Print per-class metrics.
    print()
    print('Per-class results:')
    print('Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    print('all\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_aps['all'],
                    class_tps['all']['trans_err'],
                    class_tps['all']['scale_err'],
                    class_tps['all']['orient_err'],
                    class_tps['all']['vel_err'],
                    class_tps['all']['attr_err']))

if __name__ == "__main__":
    main()
    
