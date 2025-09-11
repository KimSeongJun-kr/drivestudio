import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, OrderedDict


# nuScenes category_name -> detection_name 매핑 (convert_prep_to_pred.py 참고)
DETECTION_MAPPING: Dict[str, str] = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def read_annotation_file(ann_file_path: str) -> List[Dict[str, Any]]:
    """annotation JSON을 읽어서 딕셔너리 리스트로 반환합니다.
    
    Args:
        ann_file_path: annotation 파일 경로
        
    Returns:
        List[Dict[str, Any]]: annotation 레코드 딕셔너리 리스트
    """
    with open(ann_file_path, 'r') as f:
        annotations = json.load(f)
    # 파일 형식이 리스트라고 가정 (nuScenes sample_annotation.json)
    if not isinstance(annotations, list):
        raise ValueError("annotation 파일 형식이 리스트가 아닙니다.")
    return annotations


def load_attribute_map(nusc_raw_root: str) -> Dict[str, str]:
    """nuScenes attribute.json에서 token->name 매핑을 로드합니다.

    Args:
        nusc_raw_root: nuScenes raw JSON들이 위치한 루트 (예: /path/to/v1.0-mini)

    Returns:
        Dict[str, str]: attribute_token -> attribute_name 매핑
    """
    attr_json = os.path.join(nusc_raw_root, "attribute.json")
    if not os.path.exists(attr_json):
        # 파일이 없으면 빈 매핑 반환
        return {}
    with open(attr_json, 'r') as f:
        attrs = json.load(f)
    return {rec.get("token", ""): rec.get("name", "") for rec in attrs if rec.get("token")}


def write_prediction_file(
    ann_list: List[Dict[str, Any]],
    output_path: str,
    attribute_map: Dict[str, str],
    non_gt_pair_set: Optional[Set[Tuple[str, str]]],
) -> None:
    """annotation 딕셔너리 리스트를 NuScenes prediction 포맷의 JSON 파일로 저장합니다.
    
    Args:
        ann_list: annotation 레코드 딕셔너리 리스트
        output_path: 저장할 파일 경로
    """
    # 기본 meta 정보
    meta: Dict[str, Any] = {
        "use_camera": False,
        "use_lidar": True,
    }

    # sample_token별로 그룹화
    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for ann in ann_list:
        sample_token = str(ann.get("sample_token", ""))
        if not sample_token:
            continue

        # attribute_name 설정 (간단 규칙)
        attribute_tokens = ann.get("attribute_tokens", []) or []
        attr_count = len(attribute_tokens)
        if attr_count == 0:
            attribute_name = ""
        elif attr_count == 1:
            attribute_name = attribute_map.get(attribute_tokens[0], "")
        else:
            raise Exception("Error: GT annotations must not have more than one attribute!")

        # detection_name 매핑 (없으면 car)
        category_name = ann.get("category_name", "")
        detection_name = DETECTION_MAPPING.get(str(category_name), "car")

        translation = [float(x) for x in ann.get("translation", [0.0, 0.0, 0.0])]
        size = [float(x) for x in ann.get("size", [0.0, 0.0, 0.0])]
        rotation = [float(x) for x in ann.get("rotation", [1.0, 0.0, 0.0, 0.0])]
        instance_token = str(ann.get("instance_token", ""))

        # 겹침 여부에 따른 gt_data 결정
        if non_gt_pair_set is not None:
            is_overlap = (instance_token, sample_token) in non_gt_pair_set
            gt_data = 0 if is_overlap else 1
        else:
            gt_data = -1

        prediction = OrderedDict([
            ("sample_token", sample_token),
            ("translation", translation),
            ("size", size),
            ("rotation", rotation),
            ("velocity", [0.0, 0.0]),  # annotation에는 velocity 정보가 없음
            ("detection_name", detection_name),
            ("detection_score", 0.5),  # 기본값
            ("attribute_name", attribute_name),
            ("instance_token", instance_token),
            ("instance_idx", -1),
            ("num_gaussians", -1),
            ("gt_data", gt_data),
        ])

        results[sample_token].append(prediction)

    # 최종 prediction 파일 구조
    prediction_data: Dict[str, Any] = {
        "meta": meta,
        "results": dict(results),
    }

    with open(output_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes annotation JSON to prediction format")
    parser.add_argument(
        "--ann_file",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/raw/v1.0-mini/sample_annotation_centerpoint.json",
        help="Path to annotation json file (sample_annotation.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prediction file path (optional, default: same directory as input with _pred.json suffix)",
    )
    parser.add_argument(
        "--nusc_raw_root",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/raw/v1.0-mini",
        help="nuScenes raw JSON 루트 경로 (attribute.json 등)",
    )
    parser.add_argument(
        "--non_gt_check_pred_file",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/raw/v1.0-mini/sample_annotation_centerpoint_only_matched_pred.json",
        help="nuScenes raw JSON 루트 경로 (attribute.json 등)",
    )

    args = parser.parse_args()

    # 출력 파일 경로 설정
    if args.output is None:
        input_path = Path(args.ann_file)
        output_path = input_path.parent / (input_path.stem + "_pred.json")
    else:
        output_path = Path(args.output)

    # annotation 파일 읽기
    print(f"annotation 파일을 읽는 중: {args.ann_file}")
    annotations = read_annotation_file(args.ann_file)
    print(f"총 {len(annotations)}개의 annotation을 읽었습니다.")

    # attribute 매핑 로드
    attribute_map = load_attribute_map(args.nusc_raw_root)

    # 비교용 prediction 파일에서 (instance_token, sample_token) 쌍 수집
    non_gt_pair_set = None
    if args.non_gt_check_pred_file is not None:
        non_gt_pair_set: Set[Tuple[str, str]] = set()
        with open(args.non_gt_check_pred_file, 'r') as f:
            pred_ref = json.load(f)
        # prediction 포맷 가정: {"meta":..., "results": {sample_token: [ {instance_token,...}, ... ], ...}}
        results = pred_ref.get("results", {}) if isinstance(pred_ref, dict) else {}
        for sample_token, preds in results.items():
            if not isinstance(preds, list):
                continue
            for p in preds:
                instance_token = str(p.get("instance_token", ""))
                if instance_token != "":
                    non_gt_pair_set.add((instance_token, str(sample_token)))

    # prediction 파일로 변환하여 저장
    print(f"prediction 파일로 변환 중: {output_path}")
    write_prediction_file(annotations, str(output_path), attribute_map, non_gt_pair_set)
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    main()
