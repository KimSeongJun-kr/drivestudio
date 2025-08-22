import argparse
import json
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass
from collections import defaultdict, OrderedDict


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


def read_annotation_file(ann_file_path: str) -> List[Annotation]:
    """annotation 파일을 읽어서 Annotation 객체 리스트로 반환합니다.
    
    Args:
        ann_file_path: annotation 파일 경로
        
    Returns:
        List[Annotation]: Annotation 객체 리스트
    """
    with open(ann_file_path, 'r') as f:
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


def write_prediction_file(ann_list: List[Annotation], output_path: str) -> None:
    """Annotation 객체 리스트를 NuScenes prediction 포맷의 JSON 파일로 저장합니다.
    
    Args:
        ann_list: Annotation 객체 리스트
        output_path: 저장할 파일 경로
    """
    # 기본 meta 정보
    meta = {
        "use_camera": False,
        "use_lidar": True
    }
    
    # sample_token별로 그룹화
    results = defaultdict(list)
    
    for ann in ann_list:
        # attribute_name 설정 (임시값)
        attribute_name = "vehicle.moving" if ann.attribute_tokens else ""
            
        # detection_name 설정 (임시값 - 일반적으로 가장 많은 클래스)
        detection_name = "car"
        
        prediction = OrderedDict([
            ("sample_token", ann.sample_token),
            ("translation", [float(x) for x in ann.translation]),
            ("size", [float(x) for x in ann.size]),
            ("rotation", [float(x) for x in ann.rotation]),
            ("velocity", [0.0, 0.0]),  # 기본값 - annotation에는 velocity 정보가 없음
            ("detection_name", detection_name),
            ("detection_score", 0.5),  # 기본값
            ("attribute_name", attribute_name)
        ])
        
        results[ann.sample_token].append(prediction)
    
    # 최종 prediction 파일 구조
    prediction_data = {
        "meta": meta,
        "results": dict(results)
    }
    
    with open(output_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NuScenes annotation JSON to prediction format")
    parser.add_argument(
        "--ann_file",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise/mini/001/instances/instances_info_noisy.json",
        help="Path to annotation json file (sample_annotation.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output prediction file path (optional, default: same directory as input with _pred.json suffix)",
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

    # prediction 파일로 변환하여 저장
    print(f"prediction 파일로 변환 중: {output_path}")
    write_prediction_file(annotations, str(output_path))
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    main()
