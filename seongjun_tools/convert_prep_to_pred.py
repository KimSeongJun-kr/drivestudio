import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="프리프로세스 instances_info(_noisy).json을 NuScenes prediction 포맷으로 변환합니다.")
    parser.add_argument(
        "--prep_file",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini/001/instances/instances_info.json",
        # default="/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini/001/instances/instances_info_noisy.json",
        help="입력 instances_info(_noisy).json 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 prediction 파일 경로 (미지정 시 입력 파일과 동일 폴더에 _pred.json 접미사 생성)",
    )
    parser.add_argument(
        "--nusc_raw_root",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/raw/v1.0-mini",
        help="nuScenes raw JSON들이 위치한 루트 (instance.json, sample_annotation.json 등)",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.5,
        help="detection_score 기본값",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_4x4(matrix_like: Any) -> np.ndarray:
    arr = np.asarray(matrix_like, dtype=np.float64)
    if arr.size == 16:
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"obj_to_world가 4x4가 아닙니다. shape={arr.shape}")
    return arr


def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    # 반환 형식: [w, x, y, z]
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    # 정규화
    q = q / (np.linalg.norm(q) + 1e-12)
    return q.astype(float).tolist()


def convert_instances_to_predictions(
    instances_info: Dict[str, Any],
    instance_to_sample_tokens: Dict[str, List[str]],
    score: float,
) -> Dict[str, Any]:
    meta = {"use_camera": False, "use_lidar": True}
    results: Dict[str, List[Dict[str, Any]]] = {}

    for instance_idx, inst in instances_info.items():
        instance_token = inst.get("id", "")
        class_name = inst.get("class_name", "")
        frame_ann: Dict[str, Any] = inst.get("frame_annotations", {})
        frame_idx_list: List[int] = frame_ann.get("frame_idx", [])  # type: ignore
        obj_to_world_list = frame_ann.get("obj_to_world", [])
        box_size_list: List[List[float]] = frame_ann.get("box_size", [])  # [w,l,h]

        # 실제 instance_token은 inst의 id
        inst_token = inst.get("id", "")
        if not inst_token:
            raise ValueError("instances_info 내 인스턴스에 'id'가 없습니다.")

        # 입력 데이터 정합성 검사
        num_frames = len(frame_idx_list)
        if not (len(obj_to_world_list) == num_frames and len(box_size_list) == num_frames):
            raise ValueError(
                f"인스턴스 {inst_token}의 frame_annotations 길이가 일치하지 않습니다. "
                f"frame_idx={num_frames}, obj_to_world={len(obj_to_world_list)}, box_size={len(box_size_list)}"
            )

        # nuScenes 체인 매핑 존재 여부
        if inst_token not in instance_to_sample_tokens:
            raise KeyError(f"raw instance.json / sample_annotation.json에서 인스턴스 토큰을 찾을 수 없습니다: {inst_token}")
        sample_tokens_ordered = instance_to_sample_tokens[inst_token]
        L_tokens = len(sample_tokens_ordered)

        # 5프레임 간격 오프셋 정렬: 0..4 중에서 체인 길이와 일치하는 오프셋을 우선 선택
        offset_to_indices: List[Tuple[int, List[int]]] = []
        for off in range(5):
            idxs = [i for i, fi in enumerate(frame_idx_list) if (int(fi) - off) % 5 == 0]
            offset_to_indices.append((off, idxs))

        # 정확히 일치하는 오프셋 우선
        chosen_indices: Optional[List[int]] = None
        for off, idxs in offset_to_indices:
            if len(idxs) == L_tokens:
                chosen_indices = idxs
                break
        # 없으면 길이가 더 긴 오프셋 중에서 앞 L_tokens만 사용
        if chosen_indices is None:
            for off, idxs in offset_to_indices:
                if len(idxs) > L_tokens:
                    chosen_indices = idxs[:L_tokens]
                    break
        if chosen_indices is None:
            raise ValueError(
                f"인스턴스 {inst_token}: 5프레임 간격으로 선택한 프레임 수가 sample 체인 길이({L_tokens})에 미달합니다. "
                f"candidates={[len(x[1]) for x in offset_to_indices]}"
            )

        for k in range(L_tokens):
            sel = chosen_indices[k]
            frame_idx = frame_idx_list[sel]
            otw = obj_to_world_list[sel]
            size_wlh = box_size_list[sel]
            try:
                T = parse_4x4(otw)
            except Exception:
                # 스킵
                continue
            R = T[:3, :3]
            t = T[:3, 3]
            quat = rotation_matrix_to_quaternion(R)

            sample_token = sample_tokens_ordered[k]
            if not sample_token:
                raise ValueError(f"인스턴스 {inst_token}의 k={k} 위치에서 sample_token이 비어있습니다.")

            det_name = DETECTION_MAPPING.get(class_name, "")

            pred = {
                "sample_token": sample_token,
                "translation": [float(t[0]), float(t[1]), float(t[2])],
                "size": [float(size_wlh[1]), float(size_wlh[0]), float(size_wlh[2])],
                "rotation": [float(q) for q in quat],  # [w,x,y,z]
                "velocity": [0.0, 0.0],
                "detection_name": det_name,
                "detection_score": float(score),
                "attribute_name": "",
                "instance_token": instance_token,
                "instance_idx": int(instance_idx),
                "num_gaussians": -1,
            }

            if sample_token not in results:
                results[sample_token] = []
            results[sample_token].append(pred)

    return {"meta": meta, "results": results}


def _load_nuscenes_raw_maps(raw_root: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    instance_json = os.path.join(raw_root, "instance.json")
    sample_ann_json = os.path.join(raw_root, "sample_annotation.json")

    with open(instance_json, "r") as f:
        instances = json.load(f)
    with open(sample_ann_json, "r") as f:
        sample_anns = json.load(f)

    instance_first: Dict[str, str] = {}
    for rec in instances:
        token = rec.get("token")
        first = rec.get("first_annotation_token", "")
        if token:
            instance_first[token] = first

    sample_ann_map: Dict[str, Dict[str, Any]] = {rec["token"]: rec for rec in sample_anns}
    return instance_first, sample_ann_map


def _build_instance_to_sample_tokens(
    instance_first: Dict[str, str],
    sample_ann_map: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    instance_to_samples: Dict[str, List[str]] = {}
    # 역맵: annotation token -> instance token (optional, but we can traverse via first->next)
    # 여기서는 각 instance에 대해 체인 따라가며 sample_token 수집
    for inst_token, first_ann in instance_first.items():
        curr = first_ann
        ordered_tokens: List[str] = []
        visited = set()
        while curr and curr in sample_ann_map and curr not in visited:
            visited.add(curr)
            rec = sample_ann_map[curr]
            ordered_tokens.append(rec.get("sample_token", ""))
            curr = rec.get("next", "")
        instance_to_samples[inst_token] = [t for t in ordered_tokens if t]
    return instance_to_samples


def main() -> None:
    args = parse_args()

    # 출력 경로 설정
    if args.output is None:
        input_path = Path(args.prep_file)
        output_path = input_path.parent / (input_path.stem + "_pred.json")
    else:
        output_path = Path(args.output)

    print(f"입력 프리프로세스 파일: {args.prep_file}")
    data = load_json(args.prep_file)

    print("nuScenes raw 맵 로딩 중...")
    instance_first, sample_ann_map = _load_nuscenes_raw_maps(args.nusc_raw_root)
    instance_to_sample_tokens = _build_instance_to_sample_tokens(instance_first, sample_ann_map)

    print("prediction 포맷으로 변환 중...")
    prediction = convert_instances_to_predictions(
        instances_info=data,
        instance_to_sample_tokens=instance_to_sample_tokens,
        score=args.score,
    )

    save_json(prediction, str(output_path))
    print(f"변환 완료: {output_path}")


if __name__ == "__main__":
    main()


