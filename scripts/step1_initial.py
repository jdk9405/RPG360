import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from time import time
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    BackprojectDepth, NormalSurface,
    get_intrinsics, get_cam_pts, get_transformed_cube_normal,
    cubemap_scale_alignment, get_confidence_map, get_ERP_from_cubemap,
    visualize_cube_pcds,
    parse_train_config,
)
from src.refine.refinement import refine_depth_ERP, get_refine_params
from src.camera.Projection import Cube2Equirec, EquirecGrid, Equirec2Cube


def parse_args():
    parser = argparse.ArgumentParser(description="DNR")
    parser.add_argument("--file", type=str, default="configs/mp3d_test.yaml", help="configuration file")
    args = parser.parse_args()
    assert args.file.endswith(".yaml")
    return args


def load_image_names(cfg):
    if cfg.datasets.split != "":
        with open(cfg.datasets.split, "r") as f:
            names = [
                line.strip().split()[0]
                for line in f
                if line.strip().split()[0].split(".")[-1].lower().endswith("g")
            ]
    else:
        names = [
            name for name in os.listdir(cfg.datasets.root_dir)
            if name.split(".")[-1].lower().endswith("g")
        ]
    return names


def build_model(device):
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2", pretrain=True)
    model.to(device).eval()
    return model


def prepare_cube(image, device):
    """Convert ERP image to cubemap and set up metric3d-specific inputs."""
    batch = torch.FloatTensor(image.transpose(2, 0, 1)[None])
    e2c = Equirec2Cube(384, image.shape[0])
    cube = e2c(batch).to(device)  # [6, 3, 384, 384]

    input_size = (616, 1064)
    pad_h = input_size[0] - 384
    pad_w = input_size[1] - 384
    extras = {
        "e2c": e2c,
        "pad_info": [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2],
        "mean": torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None],
        "std": torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None],
    }
    return cube, extras


def predict_depth_normal(cube, model, pad_info, mean, std, device, chunk_size=3):
    """Run metric3d inference on cube faces in chunks to balance speed and VRAM usage."""
    pad_h0, pad_h1, pad_w0, pad_w1 = pad_info
    start = time()

    # Normalize then pad with 0 (equivalent to padding with mean values before normalizing).
    # cube: [6, 3, 384, 384], values 0-255
    cube_norm = torch.div(cube.float() - mean.to(device), std.to(device))  # [6, 3, 384, 384]
    # F.pad order: (left, right, top, bottom)
    cube_padded = F.pad(cube_norm, (pad_w0, pad_w1, pad_h0, pad_h1), value=0.0)  # [6, 3, 616, 1064]

    depth_chunks, normal_chunks = [], []
    with torch.no_grad():
        for chunk in cube_padded.split(chunk_size, dim=0):
            pred_depth, confidence, output_dict = model.inference({"input": chunk})

            pred_depth = pred_depth[:, :, pad_h0: pred_depth.shape[2] - pad_h1,
                                           pad_w0: pred_depth.shape[3] - pad_w1]
            pred_normal = output_dict["prediction_normal"][:, :3, :, :]
            pred_normal = pred_normal[:, :, pad_h0: pred_normal.shape[2] - pad_h1,
                                             pad_w0: pred_normal.shape[3] - pad_w1]
            depth_chunks.append(pred_depth)
            normal_chunks.append(pred_normal)

    print(f"prediction time: {time() - start:.3f}s")
    return torch.cat(depth_chunks, dim=0), torch.cat(normal_chunks, dim=0)


def save_outputs(cfg, image_path, equi_image, equi_depth, equi_normal, equi_confidence):
    out = cfg.save.output_dir
    base = os.path.basename(image_path)
    base_png = base.replace(".jpg", ".png")

    for subdir in ["image", "init_depth", "init_normal", "confidence"]:
        os.makedirs(os.path.join(out, subdir), exist_ok=True)

    cv2.imwrite(
        os.path.join(out, "image", base),
        equi_image[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1].astype(np.uint8),
    )
    cv2.imwrite(
        os.path.join(out, "init_depth", base_png),
        (equi_depth[0][0].detach().cpu().numpy() * 1000.0).astype(np.uint16),
    )
    cv2.imwrite(
        os.path.join(out, "init_normal", base_png),
        ((equi_normal[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 65535).astype(np.uint16),
    )
    cv2.imwrite(
        os.path.join(out, "confidence", base_png),
        (equi_confidence[0][0].detach().cpu().numpy() * 65535).astype(np.uint16),
    )

    # Save sky mask for loc360 query data
    if "loc360" in out and "query_360" in out:
        sky_mask_name = os.path.join(
            cfg.datasets.root_dir, "preprocess_masks",
            os.path.basename(image_path).replace("/image/", "/masks/").replace(".jpg", ".npz"),
        )
        if os.path.exists(sky_mask_name):
            os.makedirs(os.path.join(out, "mask"), exist_ok=True)
            sky_mask = np.load(sky_mask_name)["arr_0"][0, 0]
            equi_mask = cv2.resize(1 - sky_mask, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out, "mask", base), (equi_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = parse_args()
    cfg = parse_train_config("configs/default_config", args.file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = load_image_names(cfg)

    print("[Process] Building model ...")
    model = build_model(device)
    print("[Process] Done!")

    for name in tqdm(names):
        image_path = os.path.join(cfg.datasets.root_dir, name)

        if os.path.exists(os.path.join(cfg.save.output_dir, "init_depth", os.path.basename(image_path))):
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        if w // 2 != h:
            image = cv2.resize(image, (h * 2, h), interpolation=cv2.INTER_LINEAR)

        cube, extras = prepare_cube(image, device)
        e2c = extras["e2c"]

        cube_depth, cube_normal = predict_depth_normal(
            cube, model, extras["pad_info"],
            extras["mean"], extras["std"], device)
        cube_normal = cube_normal / cube_normal.norm(dim=1, keepdim=True)

        cube_R = torch.from_numpy(np.stack(e2c.R_lst, axis=0)).to(device)
        K, inv_K = get_intrinsics(device=device)
        backproject_depth = BackprojectDepth(batch_size=6, height=384, width=384).to(device)

        cube_cam_pts = get_cam_pts(backproject_depth, cube_depth, inv_K)
        cube_normal, cube_normal_from_depth = get_transformed_cube_normal(
            cube_cam_pts, cube_normal, cube_R, device)

        # Step 1: initial scaling
        rescaled_cube_depth = cubemap_scale_alignment(
            cube_depth, cube_cam_pts, cube_normal,
            real_height=cfg.datasets.real_height, device=device)

        cube_confidence = get_confidence_map(cube_normal, cube_normal_from_depth,
                                             min_conf=cfg.datasets.min_conf)
        loss_param, opt_param = get_refine_params(cfg)

        equi_image, equi_depth, equi_normal, equi_normal_from_depth, equi_confidence, equi_mask = \
            get_ERP_from_cubemap(
                cube, rescaled_cube_depth,
                cube_normal, cube_confidence,
                cube_length=384, equi_h=512,
                device=device)

        if cfg.save.output_dir:
            save_outputs(cfg, image_path, equi_image, equi_depth, equi_normal, equi_confidence)
