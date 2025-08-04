from omegaconf import OmegaConf
import os

import viser
import nerfview
import torch
from typing import Tuple
import threading

import sys
sys.path.append('/workspace/drivestudio')
from models.gaussians.basics import dataclass_camera
from models.gaussians.basics import dataclass_gs
from models.trainers.base import BasicTrainer, GSModelType
from datasets.base.scene_dataset import SceneDataset

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

from gsplat.rendering import rasterization

import argparse
import time

class viewer_trainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.curr_frame = 0
        self.models = {}
        self.gaussian_classes = {}
        self._init_models()
        self.tic = time.time()

    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)

    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            expand_camtoworlds=None,
            expand_camtoworlds_gt=None,
            Ks=K,
            H=H,
            W=W
        )

        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in self.gaussian_classes.keys():
            if class_name in self.models and self.models[class_name] is not None:
                if class_name != "Background":
                    self.models[class_name].set_cur_frame(self.curr_frame)
                self.curr_frame += 1
                if self.curr_frame == self.num_timesteps - 1:
                    self.curr_frame = 0

                gs = self.models[class_name].get_gaussians(cam)
                if gs is None:
                    continue

                for k, _ in gs.items():
                    gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            # gs_dict[k] = torch.cat(v, dim=0)
            if v is not None and len(v) > 0:
                gs_dict[k] = torch.cat(v, dim=0)
            else:
                gs_dict[k] = None

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )
        
        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()

    def load_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict, load_only_model=True, strict=True)

    def update_viewer(self):
        num_train_rays_per_step = self.render_cfg.batch_size * 1600 * 900
        self.viewer.lock.release()
        num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
        num_train_rays_per_sec = (
            num_train_rays_per_step * num_train_steps_per_sec
        )
        # Update the viewer state.
        self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
        # Update the scene.
        self.viewer.update(self.curr_frame, num_train_rays_per_step)

        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)

        self.viewer.lock.acquire()
        self.tic = time.time()

def force_redraw_heartbeat(get_clients, hz=10, eps=1e-3):
    while True:
        for client in list(get_clients()):
            try:
                f = float(client.camera.fov)
                # 원자적으로 카메라 파라미터를 살짝 흔듭니다.
                with client.atomic():
                    client.camera.fov = f + eps
                    client.camera.fov = f
                client.flush()  # 즉시 반영
            except Exception:
                pass
        time.sleep(1.0 / hz)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    cfg.dataset = args.dataset
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
        
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ckpt_path", type=str, 
                        # default="/workspace/drivestudio/output/box_experiments_0803/iter_50k_try1_ex10_w1/checkpoint_50000_final.pth"
                        # default="/workspace/drivestudio/output/box_experiments_0803/iter_50k_try0/checkpoint_50000_final.pth"
                        # default="/workspace/drivestudio/output/box_experiments_0803/iter_600k_try0/checkpoint_100000.pth"
                        # default="/workspace/drivestudio/output/box_experiments_0801/iter_600k_try1_ex10_w1/checkpoint_100000.pth"
                        default="/workspace/drivestudio/output/box_experiments_0801/iter_600k_try0/checkpoint_200000.pth"
    )
    parser.add_argument("--config_file", help="path to config file", type=str, 
                        default="/workspace/drivestudio/configs/experiments/0803/iter_50k_try1_ex10_w1.yaml")
    parser.add_argument("--dataset", type=str, default="nuscenes/6cams_viewer")
    
    args = parser.parse_args()

    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    # dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = viewer_trainer(
        **cfg.trainer,
        num_timesteps=196,
        model_config=cfg.model,
        num_train_images=1176,
        num_full_images=1176,
        test_set_indices=[],
        scene_aabb=torch.tensor([[-100, -100, -100], [100, 100, 100]]),
        device=device
    )
    # trainer = viewer_trainer()

    trainer.load_checkpoint(args.ckpt_path)
    trainer.init_viewer()
    trainer.viewer.lock.acquire()

    # init_viewer 내부에서 만든 viser 서버 핸들을 꺼내옵니다.
    server = trainer.viewer.server  # nerfview.Viewer(server=...)로 생성된 viser 서버
    connected_clients = []

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        connected_clients.append(client)

    threading.Thread(
        target=force_redraw_heartbeat,
        args=(lambda: connected_clients, 10, 1e-4),
        daemon=True,
    ).start()

    while True:
        trainer.update_viewer()
        time.sleep(0.1)

    