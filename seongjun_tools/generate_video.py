import argparse
import torch
from omegaconf import OmegaConf
import os

import sys
sys.path.append('/workspace/drivestudio')
from models.gaussians.basics import dataclass_camera
from models.gaussians.basics import dataclass_gs
from models.trainers.scene_graph import MultiTrainer
from datasets.base.scene_dataset import SceneDataset

from datasets.driving_dataset import DrivingDataset
from tools.eval import do_evaluation

from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

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

def extract_params_from_checkpoint(ckpt_path):
    """체크포인트에서 필요한 파라미터들을 추출합니다."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # timesteps 추출 (RigidNodes의 instances_quats 첫 번째 차원)
    num_timesteps = 196  # 기본값
    if 'models/RigidNodes/instances_quats' in state_dict:
        num_timesteps = state_dict['models/RigidNodes/instances_quats'].shape[0]
    elif 'models/DeformableNodes/instances_quats' in state_dict:
        num_timesteps = state_dict['models/DeformableNodes/instances_quats'].shape[0]
    elif 'models/SMPLNodes/instances_quats' in state_dict:
        num_timesteps = state_dict['models/SMPLNodes/instances_quats'].shape[0]
    
    # 이미지 수 추출 (CamPose의 embeds.weight 첫 번째 차원)
    num_images = 1176  # 기본값
    if 'models/CamPose/embeds.weight' in state_dict:
        num_images = state_dict['models/CamPose/embeds.weight'].shape[0]
    else:
        num_images = num_timesteps * 6
    
    # scene_aabb는 체크포인트에서 직접 추출하기 어려우므로 기본값 사용
    scene_aabb = torch.tensor([[-100, -100, -100], [100, 100, 100]])
    
    return num_timesteps, num_images, scene_aabb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ckpt_path", type=str, 
                        default="/workspace/drivestudio/output/box_experiments_0910/gt_seq1_fb/checkpoint_50000_final.pth"
    )
    parser.add_argument("--config_file", help="path to config file", type=str, 
                        default="/workspace/drivestudio/configs/video.yaml")
    parser.add_argument("--dataset", type=str, 
                        default="nuscenes/6cams_video"
    )
    parser.add_argument("--scene_idx", type=int, 
                        default=1
    )
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb")
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    

    args = parser.parse_args()

    cfg = setup(args)
    args_from_cli = OmegaConf.from_cli([f"data.scene_idx={args.scene_idx}"])
    cfg = OmegaConf.merge(cfg, args_from_cli)
    print("cfg.data.scene_idx: ", cfg.data.scene_idx)

    # log_dir 설정 (eval.py에서 필요)
    if not hasattr(cfg, 'log_dir') or cfg.log_dir is None:
        # 체크포인트 경로에서 output 디렉토리 추출
        ckpt_dir = os.path.dirname(args.ckpt_path)
        cfg.log_dir = ckpt_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.init()
    _ = torch.empty(1, device=device)
    
    # 체크포인트에서 파라미터 추출
    num_timesteps, num_images, scene_aabb = extract_params_from_checkpoint(args.ckpt_path)

    # 이미지 해상도 추출 (설정 파일에서 또는 기본값 사용)
    image_height = cfg.data.get('load_size', [900, 1600])[0] if 'data' in cfg and 'load_size' in cfg.data else 900
    image_width = cfg.data.get('load_size', [900, 1600])[1] if 'data' in cfg and 'load_size' in cfg.data else 1600

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = MultiTrainer(
        **cfg.trainer,
        num_timesteps=num_timesteps,
        model_config=cfg.model,
        num_train_images=num_images,
        num_full_images=num_images,
        test_set_indices=[],
        # scene_aabb=scene_aabb,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        # image_height=image_height,
        # image_width=image_width,
        device=device
    )
    trainer.resume_from_checkpoint(args.ckpt_path)


    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_exp_lidar:
        render_keys.insert(0, "lidar_expand_on_depts")
        render_keys.insert(0, "lidar_expand_on_images")
        render_keys.insert(0, "expand_depths")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    do_evaluation(
        step=-1,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )
    