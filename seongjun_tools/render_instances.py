import os
import argparse
import numpy as np  # type: ignore
import torch  # type: ignore

from typing import Dict, Tuple, Optional, cast

# 로컬 패키지 경로 추가 (프로젝트 루트 기준으로 동작하게 함)
import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from seongjun_tools.web_viewer import (
    viewer_trainer,
    setup as setup_from_cfg,
    extract_params_from_checkpoint,
)

from models.gaussians.basics import dataclass_camera
from gsplat.rendering import rasterization  # type: ignore
from gsplat.cuda._wrapper import spherical_harmonics  # type: ignore
from utils.geometry import transform_points
from datasets.driving_dataset import DrivingDataset


def compute_topview_camera_for_instance(
    means_local: torch.Tensor,
    image_size: Tuple[int, int],
    device: torch.device,
    margin: float = 1.2,
    distance_scale: float = 3.0,
    up_axis: str = "z",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    객체 로컬 좌표계 기준 탑뷰 카메라 파라미터(K, camtoworld)를 계산합니다.

    - 카메라는 R=I, t=[0,0,d]로 두고 -Z 방향을 바라보도록 기본 가정(프로젝트 렌더링 컨벤션과 일치)
    - FOV 대신 intrinsics(K)를 계산: r (=xy 최대 반경)을 픽셀에 정확히 맞도록 fx, fy를 설정
    """
    W, H = image_size

    # 카메라 방향/위치 설정 및 회전행렬 구성
    if up_axis.lower() == "y":
        # 인간(SMPL): up=+Y, top-view는 -Y 방향을 바라보도록 설정
        forward_world = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
        world_up_hint = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        right_world = torch.cross(world_up_hint, forward_world)
        right_world = right_world / (torch.linalg.norm(right_world) + 1e-8)
        up_world = torch.cross(forward_world, right_world)
        up_world = up_world / (torch.linalg.norm(up_world) + 1e-8)
        R_c2w = torch.stack([right_world, up_world, forward_world], dim=1)
        R_w2c = R_c2w.t()
        means_cam = (means_local @ R_w2c)  # (N,3)
        r_xy = float(torch.max(means_cam[:, :2].abs()).item()) * margin + 1e-6
        d = max(r_xy * distance_scale, 1e-2)
        # preserve aspect ratio: use same focal for x/y based on max radius
        f = (min(W, H) * 0.5) * d / r_xy
        fx = fy = f
        cx, cy = W * 0.5, H * 0.5
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        cam_pos = -forward_world * d
        camtoworld = torch.eye(4, dtype=torch.float32, device=device)
        camtoworld[:3, :3] = R_c2w
        camtoworld[:3, 3] = cam_pos
    else:
        # 차량/리짓 등: up=+Z, top-view는 +Z 방향을 바라보도록 설정
        forward_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        world_up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        right_world = torch.cross(world_up_hint, forward_world)
        right_world = right_world / (torch.linalg.norm(right_world) + 1e-8)
        up_world = torch.cross(forward_world, right_world)
        up_world = up_world / (torch.linalg.norm(up_world) + 1e-8)
        R_c2w = torch.stack([right_world, up_world, forward_world], dim=1)
        R_w2c = R_c2w.t()
        means_cam = (means_local @ R_w2c)
        r_xy = float(torch.max(means_cam[:, :2].abs()).item()) * margin + 1e-6
        d = max(r_xy * distance_scale, 1e-2)
        f = (min(W, H) * 0.5) * d / r_xy
        fx = fy = f
        cx, cy = W * 0.5, H * 0.5
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        cam_pos = -forward_world * d
        camtoworld = torch.eye(4, dtype=torch.float32, device=device)
        camtoworld[:3, :3] = R_c2w
        camtoworld[:3, 3] = cam_pos

    return K, camtoworld


def compute_oriented_camera_for_instance(
    means_local: torch.Tensor,
    image_size: Tuple[int, int],
    device: torch.device,
    *, up_axis: str, view_mode: str, margin: float = 1.2, distance_scale: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W, H = image_size
    is_y_up = up_axis.lower() == "y"
    vm = view_mode.lower()
    if vm == "top":
        forward_world = torch.tensor([0.0, -1.0, 0.0], device=device) if is_y_up else torch.tensor([0.0, 0.0, 1.0], device=device)
        world_up_hint = torch.tensor([0.0, 0.0, 1.0], device=device) if is_y_up else torch.tensor([0.0, 1.0, 0.0], device=device)
    elif vm == "front":
        forward_world = torch.tensor([0.0, 0.0, 1.0], device=device) if is_y_up else torch.tensor([1.0, 0.0, 0.0], device=device)
        world_up_hint = torch.tensor([0.0, -1.0, 0.0], device=device) if is_y_up else torch.tensor([0.0, 0.0, -1.0], device=device)
    elif vm == "side":
        forward_world = torch.tensor([1.0, 0.0, 0.0], device=device) if is_y_up else torch.tensor([0.0, 1.0, 0.0], device=device)
        world_up_hint = torch.tensor([0.0, -1.0, 0.0], device=device) if is_y_up else torch.tensor([0.0, 0.0, -1.0], device=device)
    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")

    forward_world = forward_world.to(torch.float32)
    world_up_hint = world_up_hint.to(torch.float32)
    right_world = torch.cross(world_up_hint, forward_world)
    if torch.linalg.norm(right_world) < 1e-6:
        world_up_hint = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
        right_world = torch.cross(world_up_hint, forward_world)
    right_world = right_world / (torch.linalg.norm(right_world) + 1e-8)
    up_world = torch.cross(forward_world, right_world)
    up_world = up_world / (torch.linalg.norm(up_world) + 1e-8)
    R_c2w = torch.stack([right_world, up_world, forward_world], dim=1)
    R_w2c = R_c2w.t()
    means_cam = (means_local @ R_w2c)
    r_xy = float(torch.max(means_cam[:, :2].abs()).item()) * margin + 1e-6
    d = max(r_xy * distance_scale, 1e-2)
    f = (min(W, H) * 0.5) * d / r_xy
    fx = fy = f
    cx, cy = W * 0.5, H * 0.5
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    cam_pos = -forward_world * d
    camtoworld = torch.eye(4, dtype=torch.float32, device=device)
    camtoworld[:3, :3] = R_c2w
    camtoworld[:3, 3] = cam_pos
    return K, camtoworld


def extract_lidar_points_for_instance(
    trainer: viewer_trainer,
    true_id: int,
) -> Optional[torch.Tensor]:
    """
    인스턴스의 라이다 포인트를 모든 프레임에서 수집하여 오브젝트 로컬 좌표계에서 추출합니다.
    get_init_objects() 함수와 동일한 방식으로 모든 활성 프레임에서 라이다 포인트를 수집합니다.
    
    Returns:
        torch.Tensor: (N, 3) 라이다 포인트들 (오브젝트 로컬 좌표계)
    """
    # 데이터셋에서 라이다 데이터 추출
    dataset = trainer.dataset
    if (dataset is None or 
        not hasattr(dataset, 'lidar_source') or 
        not hasattr(dataset, 'pixel_source') or
        not hasattr(dataset, 'frame_num') or
        not hasattr(dataset, 'instance_num')):
        return None
        
    # 인스턴스 정보 가져오기
    if true_id >= dataset.instance_num:
        return None
        
    pixel_source = dataset.pixel_source
    if (not hasattr(pixel_source, 'per_frame_instance_mask') or
        not hasattr(pixel_source, 'instances_pose') or
        not hasattr(pixel_source, 'instances_size')):
        return None
        
    # 라이다 레이 데이터 가져오기 함수 확인
    if not hasattr(dataset.lidar_source, 'get_lidar_rays'):
        return None
        
    # 모든 프레임에서 라이다 포인트 수집 (get_init_objects와 동일한 방식)
    all_lidar_pts = []
    o_size = pixel_source.instances_size[true_id]
    
    print(f"[INFO] Collecting lidar points for instance true_id {true_id} from {dataset.frame_num} frames...")
    active_frames = 0
    
    for fi in range(dataset.frame_num):
        # 인스턴스가 이 프레임에서 활성화되어 있는지 확인
        instance_active = pixel_source.per_frame_instance_mask[fi, true_id]
        if not instance_active:
            continue
            
        active_frames += 1
            
        # 라이다 레이 데이터 가져오기
        lidar_dict = dataset.lidar_source.get_lidar_rays(fi)
        lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
        
        # 인스턴스 포즈 가져오기
        o2w = pixel_source.instances_pose[fi, true_id]
        
        # 월드 좌표계 라이다 포인트를 오브젝트 좌표계로 변환
        w2o = torch.inverse(o2w)
        o_pts = transform_points(lidar_pts, w2o)
        
        # 바운딩 박스 내부의 포인트만 필터링
        mask = (
            (o_pts[:, 0] > -o_size[0] / 2)
            & (o_pts[:, 0] < o_size[0] / 2)
            & (o_pts[:, 1] > -o_size[1] / 2)
            & (o_pts[:, 1] < o_size[1] / 2)
            & (o_pts[:, 2] > -o_size[2] / 2)
            & (o_pts[:, 2] < o_size[2] / 2)
        )
        
        valid_pts = o_pts[mask]
        if len(valid_pts) > 0:
            all_lidar_pts.append(valid_pts)
    
    # 모든 프레임의 라이다 포인트를 결합
    if len(all_lidar_pts) > 0:
        combined_pts = torch.cat(all_lidar_pts, dim=0)
        print(f"[INFO] Instance true_id {true_id}: Collected {len(combined_pts)} lidar points from {active_frames} active frames")
        return combined_pts
    else:
        print(f"[INFO] Instance true_id {true_id}: No lidar points found in {active_frames} active frames")
        return None


@torch.no_grad()
def render_instance_view(
    trainer: viewer_trainer,
    model,  # RigidNodes | DeformableNodes | SMPLNodes
    ins_id: int,
    out_path: str,
    true_id: int = -1,
    image_size: Tuple[int, int] = (800, 800),
    point_color: Tuple[int, int, int] = (255, 0, 0),
    point_alpha: float = 1.0,
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    lidar_color: Tuple[int, int, int] = (0, 255, 0),
    lidar_alpha: float = 0.8,
    show_lidar: bool = True,
    margin: float = 1.2,
    distance_scale: float = 3.0,
    view_mode: str = "top",  # "top" | "front" | "side"
) -> None:
    """
    - 인스턴스 로컬 가우시안(미활성화 xform)으로 뷰 렌더
    - 2D로 투영된 가우시안 중심점(red) + 바운딩 박스(green) + 모든 프레임 라이다 포인트(green) 오버레이 후 저장
    """
    gaussian = model.get_instance_activated_gs_dict(ins_id)
    if gaussian is None:
        return

    device = cast(torch.device, trainer.device)
    means = gaussian["means"].to(device)              # (N,3) in object frame
    scales = gaussian["scales"].to(device)            # (N,3)
    quats = gaussian["quats"].to(device)              # (N,4)
    opacities = gaussian["opacities"].to(device)      # (N,1)
    sh_dc = gaussian["sh_dcs"].to(device)             # (N,3)
    sh_rest = gaussian["sh_rests"].to(device)         # (N,(dim_sh-1),3)

    # 카메라(K, ctow) 구성
    bbox_xy_half = None
    if hasattr(model, "instances_size"):
        size = model.instances_size[ins_id].detach().cpu().numpy().tolist()
        bbox_xy_half = (abs(size[0]) * 0.5, abs(size[1]) * 0.5)

    # SMPL은 up-axis가 Y인 경우가 일반적이므로 top-view를 -Y 방향으로 설정
    up_axis = "y" if model.__class__.__name__ == "SMPLNodes" else "z"
    if view_mode in ("top", "front", "side"):
        K, ctow = compute_oriented_camera_for_instance(
            means_local=means,
            image_size=image_size,
            device=device,
            up_axis=up_axis,
            view_mode=view_mode,
            margin=margin,
            distance_scale=distance_scale,
        )
    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")

    H, W = image_size[1], image_size[0]

    # 색상(SH) 계산: 모델 로직과 동일
    colors_sh = torch.cat((sh_dc[:, None, :], sh_rest), dim=1)
    if getattr(model, "sh_degree", 0) > 0:
        cam_pos = ctow[:3, 3]
        viewdirs = means - cam_pos[None, :]
        viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)
        n = min(getattr(model, "step", 0) // model.ctrl_cfg.sh_degree_interval, model.sh_degree)
        rgbs = spherical_harmonics(n, viewdirs, colors_sh)
        rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
    else:
        rgbs = torch.sigmoid(colors_sh[:, 0, :])

    # 렌더링
    # 렌더 구성 플래그를 안전하게 조회
    packed = bool(getattr(trainer.render_cfg, "packed", False))
    absgrad = bool(getattr(trainer.render_cfg, "absgrad", False))
    sparse_grad = bool(getattr(trainer.render_cfg, "sparse_grad", False))
    antialiased = bool(getattr(trainer.render_cfg, "antialiased", False))
    radius_clip = float(getattr(trainer.render_cfg, "radius_clip", 0.0))

    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(),
        colors=rgbs,
        viewmats=torch.linalg.inv(ctow)[None, ...],
        Ks=K[None, ...],
        width=W,
        height=H,
        packed=packed,
        absgrad=absgrad,
        sparse_grad=sparse_grad,
        rasterize_mode="antialiased" if antialiased else "classic",
        render_mode="RGB+ED",
        radius_clip=radius_clip,
    )

    # RGB 분리 및 후처리
    rgb = renders[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()  # (H,W,3)
    img = (rgb * 255.0).astype(np.uint8)

    # 2D 투영 좌표 계산 (가우시안 중심점 시각화)
    with torch.no_grad():
        cam_pos = ctow[:3, 3]
        R_c2w = ctow[:3, :3]
        x_cam = (means - cam_pos[None, :]) @ R_c2w
        z = x_cam[:, 2]
        valid = z > 1e-6
        x = x_cam[valid, 0]
        y = x_cam[valid, 1]
        z = z[valid]
        u = (K[0, 0] * (x / z) + K[0, 2]).round().long()
        v = (K[1, 1] * (y / z) + K[1, 2]).round().long()

    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    inside = (u_np >= 0) & (u_np < W) & (v_np >= 0) & (v_np < H)
    u_np = u_np[inside]
    v_np = v_np[inside]

    # 점 찍기 (3x3 점)
    point_alpha = float(np.clip(point_alpha, 0.0, 1.0))
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(u_np + du, 0, W - 1)
            vv = np.clip(v_np + dv, 0, H - 1)
            if point_alpha >= 1.0:
                img[vv, uu, :] = np.array(point_color, dtype=np.uint8)
            elif point_alpha <= 0.0:
                # alpha 0이면 그리지 않음
                continue
            else:
                base = img[vv, uu, :].astype(np.float32)
                over = np.array(point_color, dtype=np.float32)
                blended = (1.0 - point_alpha) * base + point_alpha * over
                img[vv, uu, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # 라이다 포인트 시각화 (초록색 포인트) - 모든 프레임에서 수집
    if show_lidar and true_id >= 0:
        lidar_pts = extract_lidar_points_for_instance(trainer, true_id)
        if lidar_pts is not None:
            lidar_pts = lidar_pts.to(device)
            with torch.no_grad():
                # 라이다 포인트를 카메라 좌표계로 변환
                cam_pos = ctow[:3, 3]
                R_c2w = ctow[:3, :3]
                x_cam_lidar = (lidar_pts - cam_pos[None, :]) @ R_c2w
                z_lidar = x_cam_lidar[:, 2]
                valid_lidar = z_lidar > 1e-6
                
                if valid_lidar.any():
                    x_lidar = x_cam_lidar[valid_lidar, 0]
                    y_lidar = x_cam_lidar[valid_lidar, 1]
                    z_lidar = z_lidar[valid_lidar]
                    u_lidar = (K[0, 0] * (x_lidar / z_lidar) + K[0, 2]).round().long()
                    v_lidar = (K[1, 1] * (y_lidar / z_lidar) + K[1, 2]).round().long()
                    
                    u_lidar_np = u_lidar.detach().cpu().numpy()
                    v_lidar_np = v_lidar.detach().cpu().numpy()
                    inside_lidar = (u_lidar_np >= 0) & (u_lidar_np < W) & (v_lidar_np >= 0) & (v_lidar_np < H)
                    u_lidar_np = u_lidar_np[inside_lidar]
                    v_lidar_np = v_lidar_np[inside_lidar]
                    
                    # 라이다 포인트 찍기 (2x2 포인트, 가우시안보다 작게)
                    lidar_alpha = float(np.clip(lidar_alpha, 0.0, 1.0))
                    for du in (0, 1):
                        for dv in (0, 1):
                            uu_lidar = np.clip(u_lidar_np + du, 0, W - 1)
                            vv_lidar = np.clip(v_lidar_np + dv, 0, H - 1)
                            if lidar_alpha >= 1.0:
                                img[vv_lidar, uu_lidar, :] = np.array(lidar_color, dtype=np.uint8)
                            elif lidar_alpha > 0.0:
                                base = img[vv_lidar, uu_lidar, :].astype(np.float32)
                                over = np.array(lidar_color, dtype=np.float32)
                                blended = (1.0 - lidar_alpha) * base + lidar_alpha * over
                                img[vv_lidar, uu_lidar, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # 바운딩 박스 그리기 (객체 크기 사용 가능 시)
    if bbox_xy_half is not None:
        # 원본 로컬 박스 반경 추정
        if hasattr(model, "instances_size"):
            size_vec = model.instances_size[ins_id].detach().to(device)
            hx, hy, hz = (size_vec.abs() * 0.5).tolist()
        else:
            mins = means.min(dim=0)[0]
            maxs = means.max(dim=0)[0]
            hx, hy, hz = ((maxs - mins) * 0.5).tolist()

        # 8개 코너(로컬 좌표) 생성 후 카메라로 투영 → 2D AABB로 그리기 (뷰/업축 무관 안정)
        corners_local = torch.tensor([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        ], dtype=torch.float32, device=device)
        R_c2w = ctow[:3, :3]
        x_cam_corners = (corners_local - cam_pos[None, :]) @ R_c2w
        zc = x_cam_corners[:, 2]
        valid_c = zc > 1e-6
        if valid_c.any():
            u_c = (K[0, 0] * (x_cam_corners[valid_c, 0] / zc[valid_c]) + K[0, 2]).cpu().numpy()
            v_c = (K[1, 1] * (x_cam_corners[valid_c, 1] / zc[valid_c]) + K[1, 2]).cpu().numpy()
            umin = int(np.clip(np.floor(u_c.min()), 0, W - 1))
            umax = int(np.clip(np.ceil(u_c.max()), 0, W - 1))
            vmin = int(np.clip(np.floor(v_c.min()), 0, H - 1))
            vmax = int(np.clip(np.ceil(v_c.max()), 0, H - 1))
        else:
            umin = umax = vmin = vmax = 0

        # 테두리 두께 2px
        t = 2
        img[vmin : min(vmin + t, H), umin : umax + 1, :] = np.array(bbox_color, dtype=np.uint8)
        img[max(vmax - t + 1, 0) : vmax + 1, umin : umax + 1, :] = np.array(bbox_color, dtype=np.uint8)
        img[vmin : vmax + 1, umin : min(umin + t, W), :] = np.array(bbox_color, dtype=np.uint8)
        img[vmin : vmax + 1, max(umax - t + 1, 0) : umax + 1, :] = np.array(bbox_color, dtype=np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        from PIL import Image  # type: ignore[import]
        Image.fromarray(img).save(out_path)
    except Exception:
        # PIL 미설치 시 numpy로 간단 저장 (PNG raw fallback)
        import imageio.v2 as imageio  # type: ignore[import]
        imageio.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, 
        # default="/workspace/drivestudio/output/box_experiments_0813/dev_try1_mahala/checkpoint_50000_final.pth"
        # default="/workspace/drivestudio/output/box_experiments_0813/original/checkpoint_50000_final.pth"
        default="/workspace/drivestudio/output/box_experiments_0813/original_2/checkpoint_50000_final.pth"
    )
    parser.add_argument("--config_file", type=str, 
                        default="/workspace/drivestudio/configs/experiments/0813/original.yaml")
    parser.add_argument("--dataset", type=str, default="nuscenes/6cams_viewer")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--point_alpha", type=float, default=0.2)
    parser.add_argument("--show_lidar", type=bool, default=True, help="라이다 포인트 시각화 여부 (모든 프레임에서 수집)")
    parser.add_argument("--lidar_alpha", type=float, default=0.8, help="라이다 포인트 투명도")
    parser.add_argument("--margin", type=float, default=1.2)
    parser.add_argument("--distance_scale", type=float, default=130.0)
    args = parser.parse_args()

    cfg = setup_from_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.init()
        _ = torch.empty(1, device=device)

    num_timesteps, num_images, scene_aabb = extract_params_from_checkpoint(args.ckpt_path)

    # 이미지 해상도 (cfg.data.load_size가 있으면 그걸 쓰되, 여기선 정사각/사용자 지정도 허용)
    H = args.height
    W = args.width

    trainer = viewer_trainer(
        **cfg.trainer,
        num_timesteps=num_timesteps,
        model_config=cfg.model,
        num_train_images=num_images,
        num_full_images=num_images,
        test_set_indices=[],
        scene_aabb=scene_aabb,
        image_height=H,
        image_width=W,
        device=device,
    )
    trainer.load_checkpoint(args.ckpt_path)
    trainer.set_eval()

    if args.show_lidar:
        dataset = DrivingDataset(data_cfg=cfg.data)
        trainer.dataset = dataset

    # out_dir 기본값: ckpt 폴더/box_poses/{instance_topviews, instance_frontviews, instance_sideviews}
    if args.out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        base_dir = os.path.join(ckpt_dir, "box_poses")
    else:
        base_dir = args.out_dir

    # 대상 클래스만 선택
    target_classes = [k for k in ["RigidNodes", "DeformableNodes", "SMPLNodes"] if k in trainer.models]

    for class_name in target_classes:
        model = trainer.models[class_name]
        # 인스턴스 수 확인
        if not hasattr(model, "num_instances"):
            continue
        top_out_dir = os.path.join(base_dir, "instance_topviews", class_name)
        front_out_dir = os.path.join(base_dir, "instance_frontviews", class_name)
        side_out_dir = os.path.join(base_dir, "instance_sideviews", class_name)
        for ins_id in range(model.num_instances):
            # 메타정보 (옵션)
            det_name = None
            true_id = None
            if hasattr(model, "instances_detection_name") and ins_id < len(model.instances_detection_name):
                det_name = model.instances_detection_name[ins_id]
            if hasattr(model, "instances_true_id") and ins_id < len(model.instances_true_id):
                true_id = model.instances_true_id[ins_id]

            name_parts = [f"ins{ins_id}"]
            if det_name is not None:
                name_parts.append(str(det_name))
            if true_id is not None:
                name_parts.append(str(true_id))
            fname = "_".join(name_parts) + ".png"
            # Top
            out_path_top = os.path.join(top_out_dir, fname)
            render_instance_view(
                trainer=trainer,
                model=model,
                ins_id=ins_id,
                true_id=true_id,
                out_path=out_path_top,
                image_size=(W, H),
                point_alpha=args.point_alpha,
                show_lidar=args.show_lidar,
                lidar_alpha=args.lidar_alpha,
                margin=args.margin,
                distance_scale=args.distance_scale,
                view_mode="top",
            )
            print(f"Saved: {out_path_top}")
            # Front
            out_path_front = os.path.join(front_out_dir, fname)
            render_instance_view(
                trainer=trainer,
                model=model,
                ins_id=ins_id,
                true_id=true_id,
                out_path=out_path_front,
                image_size=(W, H),
                point_alpha=args.point_alpha,
                show_lidar=args.show_lidar,
                lidar_alpha=args.lidar_alpha,
                margin=args.margin,
                distance_scale=args.distance_scale,
                view_mode="front",
            )
            print(f"Saved: {out_path_front}")
            # Side
            out_path_side = os.path.join(side_out_dir, fname)
            render_instance_view(
                trainer=trainer,
                model=model,
                ins_id=ins_id,
                true_id=true_id,
                out_path=out_path_side,
                image_size=(W, H),
                point_alpha=args.point_alpha,
                show_lidar=args.show_lidar,
                lidar_alpha=args.lidar_alpha,
                margin=args.margin,
                distance_scale=args.distance_scale,
                view_mode="side",
            )
            print(f"Saved: {out_path_side}")


if __name__ == "__main__":
    main()


