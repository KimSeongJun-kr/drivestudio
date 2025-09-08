import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud


def find_scene_by_name(nusc: NuScenes, scene_name: str) -> Dict[str, Any]:
    for scene_record in nusc.scene:
        if scene_record.get("name") == scene_name:
            return scene_record
    raise ValueError(f"Scene with name '{scene_name}' not found.")


def collect_ordered_sample_tokens(nusc: NuScenes, scene_record: Dict[str, Any]) -> List[str]:
    sample_tokens: List[str] = []
    current_token: str = scene_record["first_sample_token"]
    while current_token:
        sample_tokens.append(current_token)
        current_sample = nusc.get("sample", current_token)
        current_token = current_sample["next"]
    return sample_tokens


def normalize_quaternion(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / norm, x / norm, y / norm, z / norm


def rotation_matrix_from_wxyz_quaternion(w: float, x: float, y: float, z: float) -> np.ndarray:
    w, x, y, z = normalize_quaternion(w, x, y, z)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rotation_matrix = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return rotation_matrix


def color_from_tracking_id(tracking_id: Any) -> Tuple[float, float, float]:
    # Stable hash to [0,1] HSV then convert to RGB
    integer_hash = abs(hash(str(tracking_id)))
    hue = (integer_hash % 360) / 360.0
    saturation = 0.75
    value = 0.95
    # HSV to RGB
    i = int(hue * 6)
    f = hue * 6 - i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    i = i % 6
    if i == 0:
        r, g, b = value, t, p
    elif i == 1:
        r, g, b = q, value, p
    elif i == 2:
        r, g, b = p, value, t
    elif i == 3:
        r, g, b = p, q, value
    elif i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    return float(r), float(g), float(b)


def build_geometries_for_samples(
    results_by_sample: Dict[str, List[Dict[str, Any]]],
    ordered_sample_tokens: List[str],
    start_index: int,
    end_index: int,
    quat_order: str = "auto",
    size_order: str = "wlh",
    min_track_boxes: int = 1,
) -> Tuple[List[o3d.geometry.Geometry], Dict[Any, List[np.ndarray]], np.ndarray]:
    # First pass: collect all detections with centers, rotations, sizes and track order
    detections_in_order: List[Dict[str, Any]] = []
    detections_by_track: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for sample_order_index, sample_token in enumerate(ordered_sample_tokens[start_index : end_index + 1], start=start_index):
        for det in results_by_sample.get(sample_token, []):
            # Required fields: translation [x, y, z], size [w, l, h], rotation [w, x, y, z], tracking_id
            translation = det.get("translation", [0.0, 0.0, 0.0])
            size_arr = det.get("size", [0.0, 0.0, 0.0])
            rotation_arr = det.get("rotation", [1.0, 0.0, 0.0, 0.0])
            tracking_id = det.get("tracking_id", det.get("trackingId", det.get("track_id", None)))
            if tracking_id is None:
                # Skip if no tracking id
                continue

            center = np.array(translation, dtype=np.float64)
            # Size order handling.
            if size_order == "wlh":
                width, length, height = float(size_arr[0]), float(size_arr[1]), float(size_arr[2])
            elif size_order == "lwh":
                length, width, height = float(size_arr[0]), float(size_arr[1]), float(size_arr[2])
            else:
                # Fallback: assume wlh
                width, length, height = float(size_arr[0]), float(size_arr[1]), float(size_arr[2])
            # Open3D expects [length, width, height]
            extents_lwh = np.array([length, width, height], dtype=np.float64)
            # Interpret quaternion order.
            # Default nuScenes result format is [w, x, y, z]. Some tools export [x, y, z, w].
            if quat_order == "wxyz":
                w, x, y, z = (
                    float(rotation_arr[0]),
                    float(rotation_arr[1]),
                    float(rotation_arr[2]),
                    float(rotation_arr[3]),
                )
            elif quat_order == "xyzw":
                w, x, y, z = (
                    float(rotation_arr[3]),
                    float(rotation_arr[0]),
                    float(rotation_arr[1]),
                    float(rotation_arr[2]),
                )
            else:
                # auto: choose the mapping that yields the smaller tilt (|x| and |y| close to 0 for nuScenes boxes)
                cand_wxyz = (
                    float(rotation_arr[0]),
                    float(rotation_arr[1]),
                    float(rotation_arr[2]),
                    float(rotation_arr[3]),
                )
                cand_xyzw = (
                    float(rotation_arr[3]),
                    float(rotation_arr[0]),
                    float(rotation_arr[1]),
                    float(rotation_arr[2]),
                )
                tilt_wxyz = cand_wxyz[1] * cand_wxyz[1] + cand_wxyz[2] * cand_wxyz[2]
                tilt_xyzw = cand_xyzw[1] * cand_xyzw[1] + cand_xyzw[2] * cand_xyzw[2]
                if tilt_wxyz <= tilt_xyzw:
                    w, x, y, z = cand_wxyz
                else:
                    w, x, y, z = cand_xyzw
            rotation_matrix = rotation_matrix_from_wxyz_quaternion(w, x, y, z)

            det_obj = {
                "sample_index": sample_order_index,
                "tracking_id": tracking_id,
                "center": center,
                "extents": extents_lwh,
                "rotation_matrix": rotation_matrix,
            }
            detections_in_order.append(det_obj)
            detections_by_track[tracking_id].append(det_obj)

    if not detections_in_order:
        return [], {}, np.zeros(3, dtype=np.float64)

    # Filter tracks by minimum number of boxes
    valid_track_ids = {tid for tid, lst in detections_by_track.items() if len(lst) >= int(min_track_boxes)}
    filtered_detections: List[Dict[str, Any]] = [d for d in detections_in_order if d["tracking_id"] in valid_track_ids]
    if not filtered_detections:
        return [], {}, np.zeros(3, dtype=np.float64)

    # Anchor: first detection's center becomes origin
    anchor_center = filtered_detections[0]["center"].copy()

    # Track center history for line drawing (after anchoring)
    track_id_to_centers: Dict[Any, List[np.ndarray]] = defaultdict(list)

    # Build geometries
    geometries: List[o3d.geometry.Geometry] = []
    for det in filtered_detections:
        tracking_id = det["tracking_id"]
        color = color_from_tracking_id(tracking_id)

        adjusted_center = det["center"] - anchor_center
        track_id_to_centers[tracking_id].append(adjusted_center)

        obb = o3d.geometry.OrientedBoundingBox(
            center=adjusted_center,
            R=det["rotation_matrix"],
            extent=det["extents"],
        )
        obb.color = color
        geometries.append(obb)

    # Build line sets per track (connecting sequential centers)
    for tracking_id, centers in track_id_to_centers.items():
        if len(centers) < 2:
            continue
        points = np.vstack(centers)
        lines = [[i, i + 1] for i in range(len(centers) - 1)]
        color = np.array([list(color_from_tracking_id(tracking_id)) for _ in lines], dtype=np.float64)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(color)
        geometries.append(line_set)

    return geometries, track_id_to_centers, anchor_center


def get_lidar_top_points_in_global(nusc: NuScenes, sample_token: str) -> Optional[np.ndarray]:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        return None
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    ego_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    file_path = os.path.join(nusc.dataroot, sd_rec["filename"])
    pc = LidarPointCloud.from_file(file_path)

    # Sensor to ego
    sensor_to_ego = np.eye(4, dtype=np.float64)
    sensor_to_ego[:3, :3] = Quaternion(cs_rec["rotation"]).rotation_matrix
    sensor_to_ego[:3, 3] = np.array(cs_rec["translation"], dtype=np.float64)

    # Ego to global
    ego_to_global = np.eye(4, dtype=np.float64)
    ego_to_global[:3, :3] = Quaternion(ego_rec["rotation"]).rotation_matrix
    ego_to_global[:3, 3] = np.array(ego_rec["translation"], dtype=np.float64)

    # Chain transform: sensor -> ego -> global
    transform_mat = ego_to_global @ sensor_to_ego
    pc.transform(transform_mat)

    return pc.points.T[:, :3]

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize tracking boxes with Open3D.")
    parser.add_argument("--results_json", type=str, required=True, help="Path to tracking results JSON.")
    parser.add_argument("--nusc_root", type=str, required=True, help="Path to nuScenes dataset root.")
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version (e.g., v1.0-trainval, v1.0-mini).",
    )
    parser.add_argument("--scene_name", type=str, required=True, help="Scene name to visualize.")
    parser.add_argument("--start_index", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End sample index (inclusive). Use -1 for the last sample of the scene.",
    )
    parser.add_argument(
        "--show_axes",
        action="store_true",
        help="Show coordinate frame at origin.",
    )
    parser.add_argument(
        "--quat_order",
        type=str,
        default="auto",
        choices=["auto", "wxyz", "xyzw"],
        help="Quaternion order in results JSON (default: auto).",
    )
    parser.add_argument(
        "--size_order",
        type=str,
        default="wlh",
        choices=["wlh", "lwh"],
        help="Size order in results JSON (default: wlh).",
    )
    parser.add_argument(
        "--show_lidar",
        action="store_true",
        help="Visualize LIDAR_TOP point clouds along with boxes.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel size for downsampling lidar points (<=0 to disable).",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=3.0,
        help="Line width for bounding boxes and track lines.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help="Point size for lidar point cloud.",
    )
    parser.add_argument(
        "--min_track_boxes",
        type=int,
        default=2,
        help="Visualize only tracks with at least this many boxes in the selected range.",
    )

    args = parser.parse_args()

    # Load results JSON
    with open(args.results_json, "r") as f:
        results_payload = json.load(f)
    results_by_sample: Dict[str, List[Dict[str, Any]]] = results_payload.get("results", {})

    # Load nuScenes and get scene samples
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    scene_record = find_scene_by_name(nusc, args.scene_name)
    ordered_sample_tokens = collect_ordered_sample_tokens(nusc, scene_record)

    start_index = max(0, int(args.start_index))
    end_index = int(args.end_index)
    if end_index < 0 or end_index >= len(ordered_sample_tokens):
        end_index = len(ordered_sample_tokens) - 1
    if start_index > end_index:
        raise ValueError(f"start_index ({start_index}) must be <= end_index ({end_index}).")

    geometries, _, anchor_center = build_geometries_for_samples(
        results_by_sample=results_by_sample,
        ordered_sample_tokens=ordered_sample_tokens,
        start_index=start_index,
        end_index=end_index,
        quat_order=args.quat_order,
        size_order=args.size_order,
        min_track_boxes=args.min_track_boxes,
    )

    if not geometries:
        raise RuntimeError("No detections found in the specified range.")

    # Lidar point cloud visualization
    if args.show_lidar:
        selected_tokens = ordered_sample_tokens[start_index : end_index + 1]
        all_points: List[np.ndarray] = []
        for tok in selected_tokens:
            pts = get_lidar_top_points_in_global(nusc, tok)
            if pts is None:
                continue
            all_points.append(pts)
        if len(all_points) > 0:
            pts_concat = np.vstack(all_points).astype(np.float64)
            # Shift by anchor center to align with boxes
            pts_concat = pts_concat - anchor_center.reshape(1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_concat)
            if args.voxel_size and args.voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))
            # Set color to light gray
            colors = np.full((len(pcd.points), 3), 0.6, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)

    if args.show_axes:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    # Use Visualizer to control line width and point size
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="nuScenes Tracking Visualization", width=1280, height=720)
    for g in geometries:
        vis.add_geometry(g)
    render_option = vis.get_render_option()
    if render_option is not None:
        if args.point_size is not None and args.point_size > 0:
            render_option.point_size = float(args.point_size)
        if args.line_width is not None and args.line_width > 0:
            render_option.line_width = float(args.line_width)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()


