import os
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Union


def _parse_rpy_list(tokens: List[str]) -> Tuple[float, float, float]:
    """문자열 토큰 리스트를 (roll, pitch, yaw) 3개 float로 파싱합니다.

    지원 형식:
    - 3개의 분리 토큰: ["5", "10", "15"]
    - 단일 토큰 CSV: ["5,10,15"]
    - 단일 토큰 JSON/브라켓: ["[5, 10, 15]"]
    """
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise argparse.ArgumentTypeError("--rpy-std-deg는 3개의 값이 필요합니다 (roll, pitch, yaw)")

    values: List[str]
    if len(tokens) == 1:
        s = tokens[0].strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if "," in s:
            values = [v.strip() for v in s.split(",") if v.strip()]
        else:
            values = [v for v in s.split() if v]
    else:
        values = tokens

    if len(values) != 3:
        raise argparse.ArgumentTypeError("--rpy-std-deg는 정확히 3개의 값(roll pitch yaw)이 필요합니다")

    try:
        rpy = (float(values[0]), float(values[1]), float(values[2]))
    except Exception:
        raise argparse.ArgumentTypeError("--rpy-std-deg 값은 숫자여야 합니다 (deg)")

    return rpy


def _parse_xyz_list(tokens: List[str]) -> Tuple[float, float, float]:
    """문자열 토큰 리스트를 (x, y, z) 3개 float로 파싱합니다.

    지원 형식:
    - 3개의 분리 토큰: ["0.1", "0.2", "0.3"]
    - 단일 토큰 CSV: ["0.1,0.2,0.3"]
    - 단일 토큰 JSON/브라켓: ["[0.1, 0.2, 0.3]"]
    """
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise argparse.ArgumentTypeError("--trans-std-xyz는 3개의 값이 필요합니다 (x, y, z)")

    values: List[str]
    if len(tokens) == 1:
        s = tokens[0].strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if "," in s:
            values = [v.strip() for v in s.split(",") if v.strip()]
        else:
            values = [v for v in s.split() if v]
    else:
        values = tokens

    if len(values) != 3:
        raise argparse.ArgumentTypeError("--trans-std-xyz는 정확히 3개의 값(x y z)이 필요합니다")

    try:
        xyz = (float(values[0]), float(values[1]), float(values[2]))
    except Exception:
        raise argparse.ArgumentTypeError("--trans-std-xyz 값은 숫자여야 합니다 (m)")

    return xyz

def save_instances_info(instances_info: Dict[str, Any], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(instances_info, f, indent=2)

def sample_translation_noise(xyz_std_m: Tuple[float, float, float], xyz_bias_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    std_xyz = np.asarray(xyz_std_m, dtype=np.float64)
    bias_xyz = np.asarray(xyz_bias_m, dtype=np.float64)
    noise = np.random.normal(loc=bias_xyz, scale=std_xyz, size=(3,))
    return noise.astype(np.float64)


def sample_rotation_noise_matrix(rpy_std_deg: Tuple[float, float, float], rpy_bias_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """
    회전 노이즈 행렬을 샘플링합니다.

    - float 입력: 기존과 동일한 등방성 축-각(axis-angle) 노이즈
    - (roll, pitch, yaw) 입력: 각 축에 대해 독립 가우시안 노이즈를 주고 Rz(yaw) @ Ry(pitch) @ Rx(roll)로 합성
    """
    roll_std_deg, pitch_std_deg, yaw_std_deg = [float(v) for v in rpy_std_deg]
    roll_bias_deg, pitch_bias_deg, yaw_bias_deg = [float(v) for v in rpy_bias_deg]
    
    roll_rad = np.random.normal(loc=np.deg2rad(roll_bias_deg), scale=np.deg2rad(roll_std_deg))
    pitch_rad = np.random.normal(loc=np.deg2rad(pitch_bias_deg), scale=np.deg2rad(pitch_std_deg))
    yaw_rad = np.random.normal(loc=np.deg2rad(yaw_bias_deg), scale=np.deg2rad(yaw_std_deg))

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)
    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # ZYX(= yaw-pitch-roll) 순서의 소량 회전 합성
    return (Rz @ Ry @ Rx).astype(np.float64)


def add_noise_to_transform(transform_4x4: np.ndarray, xyz_std_m: Tuple[float, float, float], rpy_std_deg: Tuple[float, float, float], xyz_bias_m: Tuple[float, float, float] = (0.0, 0.0, 0.0), rpy_bias_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if transform_4x4.shape != (4, 4):
        raise ValueError("transform_4x4는 (4,4) 형태여야 합니다.")

    rotation = transform_4x4[:3, :3].astype(np.float64)
    translation = transform_4x4[:3, 3].astype(np.float64)

    rotation_noise = sample_rotation_noise_matrix(rpy_std_deg, rpy_bias_deg)
    translation_noise = sample_translation_noise(xyz_std_m, xyz_bias_m)

    # 월드 좌표계에서의 소량 회전/이동 노이즈 주입: R' = R_noise @ R, t' = t + noise
    rotation_noisy = rotation_noise @ rotation
    translation_noisy = translation + translation_noise

    noisy = np.eye(4, dtype=np.float64)
    noisy[:3, :3] = rotation_noisy
    noisy[:3, 3] = translation_noisy
    return noisy


def ensure_4x4_matrix(m: Any) -> np.ndarray:
    arr = np.asarray(m, dtype=np.float64)
    if arr.size == 16:
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"obj_to_world 항목이 4x4 행렬이 아닙니다. shape={arr.shape}")
    return arr


def process_instances(instances_info: Dict[str, Any], xyz_std_m: Tuple[float, float, float], rpy_std_deg: Tuple[float, float, float], xyz_bias_m: Tuple[float, float, float] = (0.0, 0.0, 0.0), rpy_bias_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, Any]:
    for instance_key, instance_val in instances_info.items():
        frame_ann = instance_val.get("frame_annotations", {})
        obj_to_world_list: List[Any] = frame_ann.get("obj_to_world", [])
        if not isinstance(obj_to_world_list, list):
            continue

        for i in range(len(obj_to_world_list)):
            m = ensure_4x4_matrix(obj_to_world_list[i])
            noisy_m = add_noise_to_transform(m, xyz_std_m, rpy_std_deg, xyz_bias_m, rpy_bias_deg)
            # obj_to_world_list[i] = noisy_m.reshape(-1).astype(float).tolist()
            obj_to_world_list[i] = noisy_m.astype(float).tolist()

    return instances_info


def discover_sequence_roots(base_dir: str) -> List[str]:
    """
    base_dir가 단일 시퀀스 루트인지 또는 상위(여러 시퀀스 포함)인지 판별하고,
    시퀀스 루트(내부에 instances/instances_info.json이 존재하는 디렉터리) 리스트를 반환합니다.
    """
    seq_info_rel = os.path.join("instances", "instances_info.json")

    # base_dir 자체가 시퀀스 루트인 경우
    if os.path.isfile(os.path.join(base_dir, seq_info_rel)):
        return [base_dir]

    # 하위 폴더를 스캔하여 시퀀스 루트 수집
    if not os.path.isdir(base_dir):
        return []

    sequence_roots: List[str] = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        candidate = os.path.join(path, seq_info_rel)
        if os.path.isfile(candidate):
            sequence_roots.append(path)

    return sequence_roots



def single_file_process(args: argparse.Namespace) -> None:
    input_path = args.input
    if args.inplace:
        output_path = input_path
    else:
        if args.output:
            output_path = args.output
        else:
            base_dir = os.path.dirname(input_path)
            output_path = os.path.join(base_dir, "instances_info_noisy.json")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    t_std: Tuple[float, float, float]
    t_std = args.trans_std_xyz
    t_bias: Tuple[float, float, float] = args.trans_bias_xyz

    # 필수 인자: (roll, pitch, yaw) 표준편차 (deg)
    r_std: Tuple[float, float, float] = args.rpy_std_deg
    r_bias: Tuple[float, float, float] = args.rpy_bias_deg

    with open(input_path, "r") as f:
        instances_info = json.load(f)
    instances_info_noisy = process_instances(instances_info, xyz_std_m=t_std, rpy_std_deg=r_std, xyz_bias_m=t_bias, rpy_bias_deg=r_bias)
    save_instances_info(instances_info_noisy, output_path)

    print(f"입력: {input_path}")
    print(f"출력: {output_path}")
    return

def multi_file_process(args: argparse.Namespace) -> None:
    # 일괄 처리 모드: data-path가 상위(예: mini) 또는 단일 시퀀스 루트일 수 있음
    seq_roots = discover_sequence_roots(args.data_path)
    if len(seq_roots) == 0:
        maybe_file = os.path.join(args.data_path, "instances", "instances_info.json")
        raise FileNotFoundError(
            "시퀀스를 찾지 못했습니다. 다음을 확인하세요:\n"
            f" - 경로 존재 여부: {args.data_path}\n"
            " - 하위 폴더 내 instances/instances_info.json 존재 여부\n"
            f" - 혹은 파일 직접 지정: --input {maybe_file}"
        )

    t_std: Tuple[float, float, float]
    t_std = args.trans_std_xyz
    t_bias: Tuple[float, float, float] = args.trans_bias_xyz

    processed = 0
    failed: List[Tuple[str, str]] = []

    for seq_root in seq_roots:
        in_path = os.path.join(seq_root, "instances", "instances_info.json")
        if args.inplace:
            out_path = in_path
        else:
            out_path = os.path.join(seq_root, "instances", "instances_info_noisy.json")

        try:
            # 필수 인자: (roll, pitch, yaw) 표준편차 (deg)
            r_std_batch: Tuple[float, float, float] = args.rpy_std_deg
            r_bias_batch: Tuple[float, float, float] = args.rpy_bias_deg

            with open(in_path, "r") as f:
                instances_info = json.load(f)
            instances_info_noisy = process_instances(instances_info, xyz_std_m=t_std, rpy_std_deg=r_std_batch, xyz_bias_m=t_bias, rpy_bias_deg=r_bias_batch)
            save_instances_info(instances_info_noisy, out_path)
            processed += 1
            print(f"[OK] {seq_root}: {os.path.relpath(out_path, seq_root)}")
        except Exception as e:
            failed.append((seq_root, str(e)))
            print(f"[FAIL] {seq_root}: {e}")

    print("")
    print(f"총 시퀀스: {len(seq_roots)}, 성공: {processed}, 실패: {len(failed)}")
    if failed:
        print("실패 목록:")
        for seq_root, msg in failed:
            print(f" - {seq_root}: {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="instances_info.json의 obj_to_world에 회전/이동 노이즈를 추가하여 동일 형식으로 저장합니다."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=False,
        # default=None,
        # default = "/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise/mini",
        default = "/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini",
        help=(
            "시퀀스 루트 경로 또는 상위(예: mini) 경로. "
            "상위 경로를 주면 하위의 각 시퀀스(000,001,...)를 자동 탐색하여 처리합니다."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        # default=None,
        # default = "/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise_bias/mini/001/instances/instances_info.json",
        help="입력 instances_info.json 파일 경로 (우선순위가 --data-path 보다 높음)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="",
        help=(
            "출력 경로 (단일 파일 처리 시에만 사용). "
            "일괄 처리 시 각 시퀀스의 instances/instances_info_noisy.json에 저장됩니다."
        ),
    )
    parser.add_argument(
        "--inplace",
        type=bool,
        default=True,
        help="원본 파일을 덮어씁니다.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="난수 시드",
    )
    parser.add_argument(
        "--trans-std-xyz",
        nargs="+",
        default=[0.15, 0.15, 0.05],
        help="이동 노이즈 표준편차 (m) [x y z]",
    )
    parser.add_argument(
        "--rpy-std-deg",
        nargs="+",
        default=[3.0, 3.0, 15.0],
        help="회전 노이즈 표준편차 (deg) [roll pitch yaw]",
    )
    parser.add_argument(
        "--trans-bias-xyz",
        nargs="+",
        default=[0.05, 0.05, -0.01],
        help="이동 노이즈 bias (m) [x y z]",
    )
    parser.add_argument(
        "--rpy-bias-deg",
        nargs="+",
        default=[0.3, 0.3, 3.0],
        help="회전 노이즈 bias (deg) [roll pitch yaw]",
    )

    args = parser.parse_args()
    # rpy-std-deg 정규화 (문자열 토큰 → (roll, pitch, yaw) 튜플)
    args.rpy_std_deg = _parse_rpy_list(args.rpy_std_deg)
    # trans-std-xyz 정규화 (문자열 토큰 → (x, y, z) 튜플)
    args.trans_std_xyz = _parse_xyz_list(args.trans_std_xyz)
    # rpy-bias-deg 정규화 (문자열 토큰 → (roll, pitch, yaw) 튜플)
    args.rpy_bias_deg = _parse_rpy_list(args.rpy_bias_deg)
    # trans-bias-xyz 정규화 (문자열 토큰 → (x, y, z) 튜플)
    args.trans_bias_xyz = _parse_xyz_list(args.trans_bias_xyz)
    if not args.input and not args.data_path:
        parser.error("--input 또는 --data-path 중 하나는 반드시 지정해야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # 단일 파일 지정 모드
    if args.input:
        single_file_process(args)
    else:
        multi_file_process(args)

if __name__ == "__main__":
    main()


