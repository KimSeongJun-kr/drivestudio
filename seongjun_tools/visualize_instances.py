import argparse
import numpy as np
import tqdm
from typing import Callable, Tuple, List, Dict
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionBox, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.utils.splits import create_splits_scenes

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

def accumulate(gt_boxes: EvalBoxes,
                pred_boxes: EvalBoxes,
                sample_ann_tokens: Dict[str, List],
                class_name: str,
                dist_fcn: Callable,
                dist_th: float,
                verbose: bool = False) -> Tuple[DetectionMetricData, Dict[str, List[Tuple[str, DetectionBox]]]]:
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
        return DetectionMetricData.no_predictions()

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
        ann_tokens = sample_ann_tokens[pred_box.sample_token]

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx
                    match_ann_token = ann_tokens[gt_idx]

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            match_pred_boxes[pred_box.sample_token].append((match_ann_token, pred_box))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
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
        return DetectionMetricData.no_predictions()

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
                               attr_err=match_data['attr_err']), match_pred_boxes
    
def read_nus_ann_file(dataroot: str, version: str) -> List[Annotation]:
    """NuScenes annotation 파일을 읽어서 Annotation 객체 리스트로 반환합니다.
    
    Args:
        dataroot: NuScenes 데이터셋 루트 경로
        version: NuScenes 버전 (e.g. 'v1.0-mini')
        
    Returns:
        List[Annotation]: Annotation 객체 리스트
    """
    ann_path = os.path.join(dataroot, version, 'sample_annotation.json')
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    
    result = []
    for ann in annotations:
        result.append(Annotation(
            token=str(ann['token']),
            sample_token=str(ann['sample_token']),
            instance_token=str(ann['instance_token']),
            visibility_token=str(ann['visibility_token']),
            attribute_tokens=[str(token) for token in ann['attribute_tokens']],
            translation=[float(x) for x in ann['translation']],
            size=[float(x) for x in ann['size']],
            rotation=[float(x) for x in ann['rotation']],
            prev=str(ann['prev']),
            next=str(ann['next']),
            num_lidar_pts=int(ann['num_lidar_pts']),
            num_radar_pts=int(ann['num_radar_pts'])
        ))
    return result

def write_nus_ann_file(updated_ann: List[Annotation], output_path: str) -> None:
    """수정된 Annotation 객체 리스트를 NuScenes 포맷의 JSON 파일로 저장합니다.
    
    Args:
        updated_ann: 수정된 Annotation 객체 리스트
        output_path: 저장할 파일 경로
    """
    
    annotations = []
    for ann in updated_ann:
        ann_dict = OrderedDict([
            ('token', ann.token),
            ('sample_token', ann.sample_token),
            ('instance_token', ann.instance_token),
            ('visibility_token', ann.visibility_token),
            ('attribute_tokens', ann.attribute_tokens),
            ('translation', [float(x) for x in ann.translation]),
            ('size', [float(x) for x in ann.size]),
            ('rotation', [float(x) for x in ann.rotation]),
            ('prev', ann.prev),
            ('next', ann.next),
            ('num_lidar_pts', ann.num_lidar_pts),
            ('num_radar_pts', ann.num_radar_pts)
        ])
        annotations.append(ann_dict)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

def update_nus_ann_file(original_ann: List[Annotation], all_matched_pred_boxes: Dict[str, List[Tuple[str, DetectionBox]]]) -> Tuple[List[Annotation], List[Annotation]]:
    """원본 annotation을 예측 결과로 업데이트합니다.
    
    Args:
        original_ann: 원본 Annotation 객체 리스트
        all_matched_pred_boxes: 매칭된 예측 박스 정보 (sample_token -> [(ann_token, pred_box)])
        
    Returns:
        Tuple[List[Annotation], List[Annotation]]: 
            - 첫 번째: 업데이트된 전체 Annotation 객체 리스트
            - 두 번째: 매칭된 예측 박스들로만 구성된 Annotation 객체 리스트
    """
    # ann_token을 키로 하는 딕셔너리 생성
    ann_dict = {ann.token: ann for ann in original_ann}
    matched_ann_list = []
    
    # 매칭된 예측 박스 정보로 annotation 업데이트
    for sample_token, matched_boxes in all_matched_pred_boxes.items():
        for ann_token, pred_box in matched_boxes:
            if ann_token in ann_dict:
                ann = ann_dict[ann_token]
                # 예측된 박스 정보로 업데이트
                ann.translation = pred_box.translation
                ann.size = pred_box.size
                ann.rotation = pred_box.rotation
                # 매칭된 annotation을 별도 리스트에 추가
                matched_ann_list.append(ann)
    
    return list(ann_dict.values()), matched_ann_list
    
def visualize_ego_translations_3d(pred_boxes: EvalBoxes, gt_boxes: EvalBoxes, save_path: str = None) -> None:
    """pred_boxes와 gt_boxes의 ego_translation을 3D로 시각화합니다.
    
    Args:
        pred_boxes: 예측 박스들
        gt_boxes: Ground truth 박스들  
        save_path: 이미지를 저장할 경로 (None이면 화면에 출력)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prediction boxes의 ego_translation 수집
    pred_translations = []
    for sample_token in pred_boxes.sample_tokens:
        for box in pred_boxes[sample_token]:
            if hasattr(box, 'translation') and box.translation is not None:
                # translation = (box.translation[0], box.translation[1], box.translation[2])
                translation = (box.translation[0], -box.translation[2], box.translation[1])
                pred_translations.append(translation)
    
    # Ground truth boxes의 ego_translation 수집  
    gt_translations = []
    center_point = gt_boxes[gt_boxes.sample_tokens[0]][0].translation
    for sample_token in gt_boxes.sample_tokens:
        for box in gt_boxes[sample_token]:
            if hasattr(box, 'translation') and box.translation is not None:
                translation = (box.translation[0] - center_point[0], box.translation[1] - center_point[1], box.translation[2] - center_point[2])
                gt_translations.append(translation)
    
    # numpy 배열로 변환
    if pred_translations:
        pred_translations = np.array(pred_translations)
        ax.scatter(pred_translations[:, 0], pred_translations[:, 1], pred_translations[:, 2], 
                  c='red', marker='o', s=20, alpha=0.6, label=f'Predictions ({len(pred_translations)})')
    
    if gt_translations:
        gt_translations = np.array(gt_translations)
        ax.scatter(gt_translations[:, 0], gt_translations[:, 1], gt_translations[:, 2],
                  c='blue', marker='^', s=20, alpha=0.6, label=f'Ground Truth ({len(gt_translations)})')
    
    # 축 라벨 설정
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)') 
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Visualization of Ego Translations\n(Predictions vs Ground Truth)')
    
    # 범례 추가
    ax.legend()
    
    # 격자 표시
    ax.grid(True, alpha=0.3)
    
    # 축 비율 조정
    if pred_translations is not None and len(pred_translations) > 0:
        all_translations = pred_translations
        if gt_translations is not None and len(gt_translations) > 0:
            all_translations = np.vstack([pred_translations, gt_translations])
    elif gt_translations is not None and len(gt_translations) > 0:
        all_translations = gt_translations
    else:
        print("시각화할 데이터가 없습니다.")
        return
        
    # 축 범위 설정
    margin = 10.0  # 여백
    x_range = [all_translations[:, 0].min() - margin, all_translations[:, 0].max() + margin]
    y_range = [all_translations[:, 1].min() - margin, all_translations[:, 1].max() + margin]
    z_range = [all_translations[:, 2].min() - margin, all_translations[:, 2].max() + margin]
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D 시각화가 {save_path}에 저장되었습니다.")
    else:
        plt.show()
    
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--pred",
        type=str,
        default="/workspace/drivestudio/output/feasibility_check/run_original_scene_0_date_0529_try_1/keyframe_instance_poses_data/all_poses.json",
        help="Path to prediction json",
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
        default=True,
        help="Verbose",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save the 3D visualization plot"
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

    pred_boxes, meta = load_prediction(args.pred, 
                                        config.max_boxes_per_sample, 
                                        DetectionBox,
                                        verbose=args.verbose)

    gt_boxes, sample_ann_tokens = load_gt(nusc, eval_set_map[args.version], DetectionBox, verbose=args.verbose)

    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
        "Samples in split doesn't match samples in predictions."
    
    # Add center distances.
    # pred_boxes = add_center_dist(nusc, pred_boxes)
    for sample_token in pred_boxes.sample_tokens:
        for box in pred_boxes[sample_token]:
            box.ego_translation = box.translation
                
    gt_boxes = add_center_dist(nusc, gt_boxes)
    
    # 3D 시각화
    print("3D 시각화를 생성하고 있습니다...")
    visualize_ego_translations_3d(pred_boxes, gt_boxes, args.save_plot)

if __name__ == "__main__":
    main()
    
