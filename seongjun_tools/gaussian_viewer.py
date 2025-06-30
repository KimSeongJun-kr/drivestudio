import os
import sys
import torch
from typing import Dict, List, Optional
import numpy as np
import argparse
import math
import imageio
from gsplat.rendering import rasterization
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.gaussians.basics import dataclass_gs, SH2RGB  # type: ignore


def _activate_quaternion(quats: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion tensors so that \|q\| = 1."""
    return quats / quats.norm(dim=-1, keepdim=True)


def _activate_scaling(scales: torch.Tensor) -> torch.Tensor:
    """Exponentiate scale parameters to ensure positivity."""
    return torch.exp(scales)


def _activate_opacity(opacities: torch.Tensor) -> torch.Tensor:
    """Squash raw opacity parameters with sigmoid to the range (0, 1)."""
    return torch.sigmoid(opacities)


def load_gaussians_from_checkpoint(
    checkpoint_path: str,
    load_classes: Optional[List[str]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, dataclass_gs]:
    """체크포인트(.pth) 파일에서 가우시안 정보만 추출하여 반환합니다.

    Args:
        checkpoint_path (str): BasicTrainer 에서 저장한 .pth 체크포인트 경로
        load_classes (Optional[List[str]], optional): 특정 class(`Background`, `RigidNodes` 등)만
            읽고 싶다면 이름 리스트를 전달합니다. None이면 모든 가우시안 클래스를 로드합니다.
        device (torch.device, optional): 텐서를 로드할 디바이스. 기본값은 CPU 입니다.

    Returns:
        Dict[str, dataclass_gs]: key 가 class 이름이고 value 가 dataclass_gs 로 구성된
            가우시안 파라미터 집합입니다. 렌더러에 바로 투입할 수 있습니다.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 파일을 찾을 수 없습니다: {checkpoint_path}")

    print(f"🔍 체크포인트 로딩: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # checkpoint 형식 검증
    if not isinstance(checkpoint, dict) or "models" not in checkpoint:
        raise ValueError("유효하지 않은 체크포인트 형식입니다. 'models' 키가 없습니다.")

    models_state = checkpoint["models"]

    gaussian_dict: Dict[str, dataclass_gs] = {}

    for class_name, state in models_state.items():
        # 특정 클래스만 로드하는 경우 필터링
        if load_classes is not None and class_name not in load_classes:
            continue

        # 가우시안 모델 판별 – 필수 키 존재 여부 확인
        required_keys = ["_means", "_scales", "_quats", "_opacities"]
        if not all(k in state for k in required_keys):
            # Sky, CamPose 등 가우시안이 아닌 모델은 건너뜀
            continue

        means = state["_means"].to(device)
        scales = state["_scales"].to(device)
        quats = state["_quats"].to(device)
        opacities = state["_opacities"].to(device)
        features_dc = state.get("_features_dc", None)

        activated_scales = _activate_scaling(scales)
        activated_quats = _activate_quaternion(quats)
        activated_opacities = _activate_opacity(opacities)

        if features_dc is not None:
            rgbs = SH2RGB(features_dc.to(device))
        else:
            rgbs = torch.zeros_like(means, device=device)

        gaussian_dict[class_name] = dataclass_gs(
            _opacities=activated_opacities,
            _means=means,
            _rgbs=rgbs,
            _scales=activated_scales,
            _quats=activated_quats,
            detach_keys=[],
            extras=None,
        )

    if not gaussian_dict:
        raise RuntimeError("체크포인트에서 가우시안 정보를 찾지 못했습니다.")

    print(f"✅ {len(gaussian_dict)} 개의 가우시안 클래스를 성공적으로 로드했습니다.")
    return gaussian_dict

# 추가: 필요한 라이브러리
import math
import imageio
from gsplat.rendering import rasterization

# --------------------------- 카메라 유틸 ---------------------------

def build_camtoworld_from_pose(pose: list, device: torch.device) -> torch.Tensor:
    """x, y, z, roll, pitch, yaw 로부터 cam-to-world 행렬 생성.

    roll / pitch / yaw 는 degree 단위(오른손 좌표계, Z-front, Y-up 기준)이며
    회전 순서는 Z(yaw) → Y(pitch) → X(roll) (즉 R = Rz * Ry * Rx) 로 가정합니다.
    """
    if len(pose) != 6:
        raise ValueError("pose 리스트는 반드시 6개의 값을 포함해야 합니다: x y z roll pitch yaw")

    x, y, z, roll_deg, pitch_deg, yaw_deg = pose
    roll, pitch, yaw = map(math.radians, [roll_deg, pitch_deg, yaw_deg])

    # 회전행렬 구성
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)],
    ], dtype=torch.float32, device=device)

    Ry = torch.tensor([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,           1, 0          ],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ], dtype=torch.float32, device=device)

    Rz = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,           0,           1],
    ], dtype=torch.float32, device=device)

    R = Rz @ Ry @ Rx  # ZYX 순서

    camtoworld = torch.eye(4, device=device)
    camtoworld[:3, :3] = R
    camtoworld[:3, 3] = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    return camtoworld


def build_K(fov_deg: float, W: int, H: int, device: torch.device) -> torch.Tensor:
    """단일 fov(수평) 로부터 3x3 intrinsic K 생성"""
    fov_rad = math.radians(fov_deg)
    fx = fy = 0.5 * W / math.tan(fov_rad / 2)
    cx, cy = W / 2, H / 2
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], device=device)
    return K

# --------------------------- 렌더링 함수 ---------------------------

def render_image(
    gaussian_dict: Dict[str, dataclass_gs],
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    """가우시안 집합을 단일 뷰에서 렌더링하고 RGB 이미지를 반환합니다."""
    # --- 가우시안 병합 ---
    means_list, quats_list, scales_list, opacities_list, rgbs_list = [], [], [], [], []
    for gs in gaussian_dict.values():
        means_list.append(gs._means.to(device))
        quats_list.append(gs._quats.to(device))
        scales_list.append(gs._scales.to(device))
        opacities_list.append(gs._opacities.squeeze().to(device))
        rgbs_list.append(gs._rgbs.to(device))

    means = torch.cat(means_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    opacities = torch.cat(opacities_list, dim=0)
    colors = torch.cat(rgbs_list, dim=0)

    # --- 렌더링 ---
    renders, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld)[None, ...],
        Ks=K[None, ...],
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="antialiased",
    )
    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0).cpu().numpy()
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8

# --------------------------- 가우시안 병합 헬퍼 ---------------------------

def merge_gaussians(gaussian_dict: Dict[str, dataclass_gs], device: torch.device):
    """gaussian_dict 를 하나의 큰 텐서 세트로 병합하여 반환"""
    means_list, quats_list, scales_list, opacities_list, rgbs_list = [], [], [], [], []
    for gs in gaussian_dict.values():
        means_list.append(gs._means.to(device))
        quats_list.append(gs._quats.to(device))
        scales_list.append(gs._scales.to(device))
        opacities_list.append(gs._opacities.squeeze().to(device))
        rgbs_list.append(gs._rgbs.to(device))

    means = torch.cat(means_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    opacities = torch.cat(opacities_list, dim=0)
    colors = torch.cat(rgbs_list, dim=0)

    return means, quats, scales, opacities, colors


def render_image_arrays(
    arrays,
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
):
    means, quats, scales, opacities, colors = arrays
    renders, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld)[None, ...],
        Ks=K[None, ...],
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="antialiased",
    )
    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)

# --------------------------- 인터랙티브 뷰어 ---------------------------


def launch_interactive_viewer(arrays, bbox, K, width, height, device, init_pose=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # 초기 pose 설정
    if init_pose is None:
        center = bbox.mean(axis=0)  # (3,)
        z_range = bbox[1, 2] - bbox[0, 2]  # depth span
        init_pose = [
            float(center[0]), float(center[1]), float(center[2] + z_range * 2.0),
            0.0, 0.0, 0.0,
        ]
    else:
        # ensure list length 6
        if len(init_pose) != 6:
            raise ValueError("init_pose must contain 6 values: x y z roll pitch yaw")
        init_pose = list(init_pose)

    # 레이아웃: 왼쪽 70% 가 이미지, 오른쪽 25% 가 컨트롤 패널
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax_img = fig.add_subplot(gs[0, 0])  # type: ignore
    ax_ctrl_base = fig.add_subplot(gs[0, 1])  # dummy, we'll clear
    ax_ctrl_base.clear()  # type: ignore
    ax_ctrl_base.set_axis_off()  # type: ignore

    camtoworld = build_camtoworld_from_pose(init_pose, device)
    img = render_image_arrays(arrays, camtoworld, K, width, height, device)
    imshow_obj = ax_img.imshow(img)  # type: ignore
    ax_img.set_axis_off()  # type: ignore

    # 슬라이더 설정 범위
    x_min, y_min, z_min = bbox.min(axis=0) * 0.25
    x_max, y_max, z_max = bbox.max(axis=0) * 0.25

    names = ["X", "Y", "Z", "ROLL", "PITCH", "YAW"]
    ranges = [ (x_min, x_max), (y_min, y_max), (z_min, z_max), (-180, 180), (-180, 180), (-180, 180) ]
    sliders = {}
    buttons_minus = {}
    buttons_plus = {}
    # reset & step size widgets placeholders
    step_trans = [1.0]  # mutable
    step_rot = [5.0]

    # vertical stacking parameters within control area [0,1] in that axes coord
    start_y = 0.9
    delta = 0.12
    for i, name in enumerate(names):
        y_pos = start_y - i * delta
        # Minus button
        btn_minus_ax = fig.add_axes([0.76, y_pos - 0.015, 0.03, 0.03])
        buttons_minus[name] = Button(btn_minus_ax, "-")  # type: ignore
        # Slider
        slider_ax = fig.add_axes([0.80, y_pos - 0.005, 0.12, 0.02], facecolor='lightgoldenrodyellow')
        sliders[name] = Slider(slider_ax, name, ranges[i][0], ranges[i][1], valinit=init_pose[i])  # type: ignore
        # Plus button
        btn_plus_ax = fig.add_axes([0.93, y_pos - 0.015, 0.03, 0.03])
        buttons_plus[name] = Button(btn_plus_ax, "+")  # type: ignore

    # 업데이트 함수
    def update_from_sliders(event=None):
        pose = [sliders[n].val for n in names]
        camtoworld = build_camtoworld_from_pose(pose, device)
        img = render_image_arrays(arrays, camtoworld, K, width, height, device)
        imshow_obj.set_data(img)
        fig.canvas.draw_idle()

    for sld in sliders.values():
        sld.on_changed(update_from_sliders)

    # 버튼 콜백
    def make_btn_callback(nm, delta_val):
        def _cb(event):
            cur = sliders[nm].val
            sliders[nm].set_val(cur + delta_val)
        return _cb

    for nm in names:
        buttons_minus[nm].on_clicked(lambda evt, n=nm: make_btn_callback(n, -(step_trans[0] if n in ["X","Y","Z"] else step_rot[0]))(evt))
        buttons_plus[nm].on_clicked(lambda evt, n=nm: make_btn_callback(n, (step_trans[0] if n in ["X","Y","Z"] else step_rot[0]))(evt))

    # ---- 추가 컨트롤: 스텝 사이즈 조정 및 리셋 ----
    from matplotlib.widgets import TextBox
    box_trans_ax = fig.add_axes([0.79, 0.05, 0.08, 0.03])
    box_rot_ax   = fig.add_axes([0.89, 0.05, 0.08, 0.03])
    txt_trans = TextBox(box_trans_ax, "ΔTrans", initial=str(step_trans[0]))  # type: ignore
    txt_rot   = TextBox(box_rot_ax, "ΔRot", initial=str(step_rot[0]))  # type: ignore

    def submit_trans(text):
        try:
            step_trans[0] = float(text)
        except ValueError:
            pass
    def submit_rot(text):
        try:
            step_rot[0] = float(text)
        except ValueError:
            pass

    txt_trans.on_submit(submit_trans)
    txt_rot.on_submit(submit_rot)

    # reset button
    reset_ax = fig.add_axes([0.82, 0.95, 0.1, 0.04])
    btn_reset = Button(reset_ax, "Reset")  # type: ignore
    def reset_cb(event):
        for n, val in zip(names, init_pose):
            sliders[n].set_val(val)
        update_from_sliders()
    btn_reset.on_clicked(reset_cb)

    plt.show()

# --------------------------- 메인 ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Renderer – 단일 뷰 이미지 생성")
    parser.add_argument("--checkpoint_path",
                      type=str,
                      default="/workspace/drivestudio/output/feasibility_check/updated/run_update_scene_1_date_0529_try_1/checkpoint_final.pth")
    parser.add_argument("--out", type=str, 
                        default=None, 
                        help="저장할 이미지 파일명")
    parser.add_argument("--pose", type=float, nargs=6, metavar=("X","Y","Z","ROLL","PITCH","YAW"),
                        default=[17.0, -15.0, 72.0,      -35.2, -7.0, 0.0],
                        help="카메라 포즈 매개변수: x y z roll pitch yaw (degrees)")
    parser.add_argument("--fov", type=float, default=60.0, help="수평 시야각(Field of view) [deg]")
    parser.add_argument("--width", type=int, default=800, help="출력 이미지 가로 크기")
    parser.add_argument("--height", type=int, default=600, help="출력 이미지 세로 크기")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 또는 cpu")
    parser.add_argument("--interactive", action="store_true", help="슬라이더 GUI 로 실시간 시점 변경")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 1) 가우시안 로드
    gaussian_dict = load_gaussians_from_checkpoint(args.checkpoint_path, device=device)

    # 병합 및 카메라 내부 파라미터 세팅
    arrays = merge_gaussians(gaussian_dict, device)
    K = build_K(args.fov, args.width, args.height, device)

    if args.interactive:
        # bbox 계산 (translation 슬라이더 범위용)
        means = arrays[0].cpu().numpy()
        bbox = np.stack([means.min(axis=0), means.max(axis=0)], axis=0)
        launch_interactive_viewer(arrays, bbox, K, args.width, args.height, device, init_pose=args.pose)
    else:
        # 2) 카메라 구축 및 렌더링
        camtoworld = build_camtoworld_from_pose(args.pose, device)
        print("📸 렌더링 중 ...")
        rgb_image = render_image_arrays(arrays, camtoworld, K, args.width, args.height, device)

        # 3) 저장
        output_path = Path(args.out) if args.out else Path(args.checkpoint_path).parent / "render.png"
        imageio.imwrite(output_path, rgb_image)
        print(f"✅ 이미지 저장 완료: {output_path}")

    # 메모리 해제
    del gaussian_dict