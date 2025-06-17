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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyquaternion import Quaternion
import open3d as o3d

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
    # 올바른 회전 변환: (rotation_matrix @ corners.T).T
    corners_rotated = (rotation_matrix @ corners_local.T).T
    
    # 평행이동 적용
    corners_world = corners_rotated + np.array(translation)
    
    return corners_world

def draw_3d_box(ax, corners, color='blue', alpha=0.3, edge_color='black'):
    """3D 박스를 그립니다.
    
    Args:
        ax: matplotlib 3D axis
        corners: 8x3 numpy array, 박스의 8개 꼭짓점
        color: 박스 면의 색상
        alpha: 투명도
        edge_color: 테두리 색상
    """
    # 12개의 면을 정의 (각 면은 4개의 꼭짓점으로 구성)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # 아래면
        [corners[4], corners[5], corners[6], corners[7]],  # 위면
        [corners[0], corners[1], corners[5], corners[4]],  # 앞면
        [corners[2], corners[3], corners[7], corners[6]],  # 뒷면
        [corners[1], corners[2], corners[6], corners[5]],  # 오른쪽면
        [corners[4], corners[7], corners[3], corners[0]]   # 왼쪽면
    ]
    
    # Poly3DCollection을 사용해서 면들을 그리기
    poly3d = [[tuple(face[j]) for j in range(len(face))] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, 
                                        facecolors=color, 
                                        linewidths=1, 
                                        edgecolors=edge_color,
                                        alpha=alpha))

# ===========================
# Open3D 시각화 유틸리티
# ===========================

# Open3D LineSet 생성을 위한 에지 인덱스 (12개)
OPEN3D_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 아래면
    [4, 5], [5, 6], [6, 7], [7, 4],  # 위면
    [0, 4], [1, 5], [2, 6], [3, 7]   # 옆면
]

# -----------------------------
# NEW: Helper for front center sphere
# -----------------------------

def create_open3d_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """지정한 중심과 색상의 구(Sphere) Mesh를 생성합니다.

    Args:
        center (np.ndarray): 중심 좌표 (3,)
        radius (float): 구의 반지름
        color (Tuple[float, float, float]): RGB 컬러 (0~1)

    Returns:
        o3d.geometry.TriangleMesh: 시각화용 Sphere Mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def create_open3d_box(corners: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.LineSet:
    """8개 꼭짓점 정보로부터 Open3D LineSet(육면체) 객체를 생성합니다.

    Args:
        corners (np.ndarray): (8, 3) 형태의 꼭짓점 좌표
        color (Tuple[float, float, float]): RGB 컬러 (0~1)

    Returns:
        o3d.geometry.LineSet: 시각화용 LineSet
    """
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(OPEN3D_BOX_LINES)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in OPEN3D_BOX_LINES])
    return line_set


# noinspection PyBroadException
def visualize_ego_translations_open3d(pred_boxes: EvalBoxes, gt_boxes: EvalBoxes, scene_name: str = None,
                                      score_threshold: float = None, save_path: str = None, max_boxes: int = -1) -> None:
    """Open3D를 이용하여 pred_boxes와 gt_boxes를 3D로 시각화합니다.

    Args:
        pred_boxes: 예측 박스들
        gt_boxes: Ground truth 박스들
        scene_name: (선택) scene 이름 (제목 표시용)
        score_threshold: (선택) score threshold (제목 표시용)
    """

    geometries = []

    # Coordinate frame 추가
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))

    pred_count, gt_count = 0, 0
    center_translation = None

    # 첫 번째 박스의 translation을 center로 설정
    for sample_token in pred_boxes.sample_tokens:
        for box in pred_boxes[sample_token]:
            if (hasattr(box, 'translation') and box.translation is not None and
                    hasattr(box, 'size') and box.size is not None and
                    hasattr(box, 'rotation') and box.rotation is not None):
                center_translation = np.array(box.translation)
                break
        if center_translation is not None:
            break

    if center_translation is None:
        print("기준이 될 박스를 찾을 수 없습니다.")
        return

    # Prediction boxes (Red)
    for sample_token in pred_boxes.sample_tokens:
        for box in pred_boxes[sample_token]:
            if max_boxes > 0 and pred_count >= max_boxes:
                break  # 개수 제한 도달
            if (hasattr(box, 'translation') and box.translation is not None and
                    hasattr(box, 'size') and box.size is not None and
                    hasattr(box, 'rotation') and box.rotation is not None):
                
                # center 기준으로 상대 위치 계산
                relative_translation = np.array(box.translation) - center_translation
                corners = get_box_corners(relative_translation, box.size, box.rotation)
                geometries.append(create_open3d_box(corners, (1.0, 0.0, 0.0)))  # Red

                # NEW: 앞면 중심점 시각화 (Red)
                # front_center = np.mean(corners[4:8], axis=0)
                front_center = (corners[1] + corners[6]) / 2 
                geometries.append(create_open3d_sphere(front_center, radius=0.1, color=(1.0, 0.0, 0.0)))

                pred_count += 1
        if max_boxes > 0 and pred_count >= max_boxes:
            break

    # Ground truth boxes (Blue)
    for sample_token in gt_boxes.sample_tokens:
        for box in gt_boxes[sample_token]:
            if max_boxes > 0 and gt_count >= max_boxes:
                break  # 개수 제한 도달
            if (hasattr(box, 'translation') and box.translation is not None and
                    hasattr(box, 'size') and box.size is not None and
                    hasattr(box, 'rotation') and box.rotation is not None):

                # center 기준으로 상대 위치 계산
                relative_translation = np.array(box.translation) - center_translation
                corners = get_box_corners(relative_translation, box.size, box.rotation)
                geometries.append(create_open3d_box(corners, (0.0, 0.0, 1.0)))  # Blue

                # NEW: 앞면 중심점 시각화 (Blue)
                # front_center = np.mean(corners[4:8], axis=0)
                front_center = (corners[1] + corners[6]) / 2 
                geometries.append(create_open3d_sphere(front_center, radius=0.1, color=(0.0, 0.0, 1.0)))

                gt_count += 1
        if max_boxes > 0 and gt_count >= max_boxes:
            break

    if pred_count == 0 and gt_count == 0:
        print("시각화할 박스가 없습니다.")
        return

    # 윈도우 이름 설정
    window_name = "Open3D Visualization of Detection Boxes"
    subtitle_parts = []
    if scene_name:
        subtitle_parts.append(f"Scene: {scene_name}")
    if score_threshold is not None and score_threshold > 0:
        subtitle_parts.append(f"Score≥{score_threshold}")
    subtitle_parts.append(f"Pred: {pred_count}, GT: {gt_count}")
    if subtitle_parts:
        window_name += " (" + " | ".join(subtitle_parts) + ")"

    # ---------------------------
    # 시각화 (온스크린 or 오프스크린)
    # ---------------------------

    # 1) 오프스크린 렌더링 모드가 필요한 경우 (save_path 지정 or GUI 사용 불가)
    if save_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"✅ 3D 시각화 결과가 '{save_path}' 로 저장되었습니다.")
        return

    # 2) 일반 윈도우 모드 (GUI 가능 환경)
    try:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    except Exception as e:
        print("⚠️ Open3D GUI 창 생성에 실패했습니다. (Headless 환경으로 판단)\n   → 오류 메시지:", e)
        print("대신 오프스크린 모드로 이미지를 저장합니다. '--save_plot <경로>' 인자를 지정하세요.")

# -----------------------------------------------------------------------------
# 박스 필터링 유틸리티 (Score / Scene)
# -----------------------------------------------------------------------------

def filter_boxes_by_score(boxes: EvalBoxes, score_threshold: float) -> EvalBoxes:
    """detection_score가 threshold 이상인 boxes만 필터링합니다.

    Args:
        boxes: 필터링할 EvalBoxes
        score_threshold: score threshold (이 값 이상인 박스들만 유지)

    Returns:
        필터링된 EvalBoxes
    """
    filtered_boxes = EvalBoxes()
    total_boxes = 0
    filtered_count = 0

    for sample_token in boxes.sample_tokens:
        sample_boxes = []
        for box in boxes[sample_token]:
            total_boxes += 1
            if hasattr(box, 'detection_score') and box.detection_score >= score_threshold:
                sample_boxes.append(box)
                filtered_count += 1

        if sample_boxes:  # 필터링된 박스가 있는 경우만 추가
            filtered_boxes.add_boxes(sample_token, sample_boxes)

    print(f"✅ Score {score_threshold} 이상인 박스: {filtered_count}/{total_boxes}개")
    return filtered_boxes


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes detection JSON to pandas DataFrame")
    parser.add_argument(
        "--pred",
        type=str,
        default="/workspace/drivestudio/output/feasibility_check/run_original_scene_0_date_0529_try_1/keyframe_instance_poses_data/all_poses.json",
        # default="/workspace/drivestudio/output/feasibility_check/run_original_scene_0_date_0529_try_1/test/results_nusc.json",
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
    parser.add_argument(
        "--scene_name",
        type=str,
        default='scene-0061',
        help="Scene name to filter boxes"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum detection score threshold for pred_boxes"
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=500,
        help="시각화할 최대 박스 개수 (<=0 이면 제한 없음)"
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
    
    # Filter pred_boxes by score
    if args.score_threshold > 0:
        print(f"📊 Score threshold {args.score_threshold}로 pred_boxes 필터링 중...")
        pred_boxes = filter_boxes_by_score(pred_boxes, args.score_threshold)
    
    # Filter boxes by scene
    if args.scene_name:
        pred_boxes = filter_boxes_by_scene(nusc, pred_boxes, args.scene_name)
        gt_boxes = filter_boxes_by_scene(nusc, gt_boxes, args.scene_name)
    
    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
        "Samples in split doesn't match samples in predictions."

    
    # Open3D 3D 시각화
    print("Open3D 3D 시각화를 생성하고 있습니다...")
    visualize_ego_translations_open3d(pred_boxes, gt_boxes, args.scene_name, args.score_threshold, args.save_plot, args.max_boxes)

if __name__ == "__main__":
    main()
    
