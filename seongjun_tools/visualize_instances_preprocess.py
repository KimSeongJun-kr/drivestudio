import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


# Open3D LineSet 생성을 위한 에지 인덱스 (12개)
OPEN3D_BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 아래면
    [4, 5], [5, 6], [6, 7], [7, 4],  # 위면
    [0, 4], [1, 5], [2, 6], [3, 7]   # 옆면
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="instances_info(원본)와 instances_info_noisy(노이즈) 박스를 Open3D로 중첩 시각화합니다."
    )
    parser.add_argument(
        "--seq-root",
        type=str,
        default="/workspace/drivestudio/data/nuscenes/drivestudio_preprocess/processed_10Hz_noise/mini/001",
        help="시퀀스 루트 경로 (내부에 instances/exists). 지정 시 --orig/--noisy가 없으면 자동으로 경로를 구성합니다.",
    )
    parser.add_argument(
        "--orig",
        type=str,
        default=None,
        help="원본 instances_info.json 경로 (미지정 시 seq-root/instances/instances_info.json)",
    )
    parser.add_argument(
        "--noisy",
        type=str,
        default=None,
        help="노이즈 instances_info_noisy.json 경로 (미지정 시 seq-root/instances/instances_info_noisy.json)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=20,
        help="시각화할 프레임 인덱스",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=1000,
        help="최대 시각화할 인스턴스 수",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="이미지로 저장할 경로 (지정 시 오프스크린 렌더링)",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="좌표축을 표시합니다.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    if args.orig is None:
        args.orig = os.path.join(args.seq_root, "instances", "instances_info.json")
    if args.noisy is None:
        candidate = os.path.join(args.seq_root, "instances", "instances_info_noisy.json")
        args.noisy = candidate if os.path.isfile(candidate) else None
    return args.orig, args.noisy


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with open(path, "r") as f:
        return json.load(f)


def parse_4x4(matrix_like: Any) -> np.ndarray:
    arr = np.asarray(matrix_like, dtype=np.float64)
    if arr.size == 16:
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"obj_to_world가 4x4 형태가 아닙니다: shape={arr.shape}")
    return arr


def get_transform_and_size_for_frame(inst: Dict[str, Any], frame_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    frame_ann = inst.get("frame_annotations", {})
    idx_list: List[int] = frame_ann.get("frame_idx", [])  # type: ignore
    otw_list = frame_ann.get("obj_to_world", [])
    size_list = frame_ann.get("box_size", [])
    try:
        ii = idx_list.index(frame_index)
    except ValueError:
        return None
    transform = parse_4x4(otw_list[ii])
    size = np.asarray(size_list[ii], dtype=np.float64)
    if size.shape != (3,):
        raise ValueError("box_size는 길이 3의 리스트여야 합니다.")
    return transform, size


def get_box_corners_from_transform(transform: np.ndarray, size_wlh: np.ndarray) -> np.ndarray:
    # size: [w, l, h]
    w, l, h = size_wlh
    corners_local = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
    ], dtype=np.float64)

    R = transform[:3, :3]
    t = transform[:3, 3]
    corners_world = (R @ corners_local.T).T + t
    return corners_world


def create_open3d_box(corners: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    # 선 두께 설정
    line_radius = 0.05
    combined_mesh = o3d.geometry.TriangleMesh()

    for line_indices in OPEN3D_BOX_LINES:
        start_point = corners[line_indices[0]]
        end_point = corners[line_indices[1]]
        line_vector = end_point - start_point
        line_length = float(np.linalg.norm(line_vector))
        if line_length < 1e-6:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=line_radius, height=line_length, resolution=8)

        z_axis = np.array([0.0, 0.0, 1.0])
        line_direction = line_vector / line_length
        rotation_axis = np.cross(z_axis, line_direction)
        norm_axis = float(np.linalg.norm(rotation_axis))

        if norm_axis > 1e-6:
            rotation_axis = rotation_axis / norm_axis
            cos_angle = float(np.dot(z_axis, line_direction))
            angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            cylinder.rotate(rotation_matrix, center=(0.0, 0.0, 0.0))

        cylinder_center = (start_point + end_point) / 2.0
        cylinder.translate(cylinder_center)
        cylinder.paint_uniform_color(color)
        combined_mesh += cylinder

    return combined_mesh


def add_boxes_geometries(
    instances_info: Dict[str, Any],
    frame_index: int,
    color: Tuple[float, float, float],
    max_instances: int,
) -> Tuple[List[o3d.geometry.TriangleMesh], int]:
    geometries: List[o3d.geometry.TriangleMesh] = []
    count = 0
    for key in sorted(instances_info.keys(), key=lambda x: int(x) if x.isdigit() else x):
        if count >= max_instances and max_instances > 0:
            break
        inst = instances_info[key]
        res = get_transform_and_size_for_frame(inst, frame_index)
        if res is None:
            continue
        transform, size = res
        corners = get_box_corners_from_transform(transform, size)
        geometries.append(create_open3d_box(corners, color))
        count += 1
    return geometries, count


def setup_window(vis: o3d.visualization.Visualizer, geometries: List, background_color: Tuple[float, float, float] = (1, 1, 1)) -> None:
    render_option = vis.get_render_option()
    render_option.background_color = np.array(background_color)
    render_option.point_size = 6.0
    for g in geometries:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.change_field_of_view(step=-500)
    ctr.set_zoom(0.2)


def render_and_save(vis: o3d.visualization.Visualizer, save_path: str) -> bool:
    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()
    try:
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        from PIL import Image
        pil_image = Image.fromarray(image_np)
        pil_image.save(save_path)
        return True
    except Exception:
        return vis.capture_screen_image(save_path)


def main() -> None:
    args = parse_args()
    orig_path, noisy_path = _resolve_paths(args)

    print(f"원본 경로: {orig_path}")
    if noisy_path is None:
        print("노이즈 파일을 찾지 못했습니다. 원본만 시각화합니다.")
    else:
        print(f"노이즈 경로: {noisy_path}")

    orig = load_json(orig_path)
    noisy = load_json(noisy_path) if noisy_path is not None else None

    geometries: List = []
    if args.show_axes:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))

    # 원본(초록), 노이즈(빨강)
    orig_geoms, orig_cnt = add_boxes_geometries(orig, args.frame_index, (0.0, 0.6, 0.0), args.max_instances)
    geometries.extend(orig_geoms)

    noisy_cnt = 0
    if noisy is not None:
        noisy_geoms, noisy_cnt = add_boxes_geometries(noisy, args.frame_index, (0.9, 0.0, 0.0), args.max_instances)
        geometries.extend(noisy_geoms)

    if len(geometries) == 0:
        print("시각화할 박스가 없습니다. frame-index를 확인하세요.")
        return

    title = f"Instances Visualization | frame={args.frame_index} | orig={orig_cnt} noisy={noisy_cnt}"

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080, window_name=title)
        setup_window(vis, geometries)
        ok = render_and_save(vis, args.save_path)
        vis.destroy_window()
        print("저장 완료" if ok else "저장 실패")
        return

    # GUI 모드
    try:
        o3d.visualization.draw_geometries(geometries, window_name=title)
    except Exception as e:
        print("GUI 모드 실패, 오프스크린으로 저장하려면 --save-path를 사용하세요.")
        print("오류:", e)


if __name__ == "__main__":
    main()


