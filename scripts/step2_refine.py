import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from time import time
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    get_confidence_map, cam_height_scale_alignment,
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


def load_inputs(cfg, name):
    out = cfg.save.output_dir
    base_png = name.replace(".jpg", ".png")

    equi_image = np.array(Image.open(os.path.join(out, "image", name))).astype(np.float32) / 255.0
    equi_depth = (cv2.imread(os.path.join(out, "init_depth", base_png), -1) / 1000.0).astype(np.float32)
    equi_normal = (cv2.imread(os.path.join(out, "init_normal", base_png), -1) / 65535.0 * 2 - 1).astype(np.float32)
    equi_confidence = (cv2.imread(os.path.join(out, "confidence", base_png), -1) / 65535.0).astype(np.float32)

    if cfg.depth_model.name in ("marigold", "geowizard"):
        equi_normal = -equi_normal

    height, width = equi_depth.shape[:2]

    mask_dir = os.path.join(out, "mask", name)
    if os.path.exists(os.path.join(out, "mask")):
        equi_mask = (cv2.imread(mask_dir, -1) / 255.0).astype(np.uint8)
        equi_mask = cv2.resize(equi_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        equi_mask = None

    equi_image = cv2.resize(equi_image, (width, height), interpolation=cv2.INTER_LINEAR)
    equi_normal = cv2.resize(equi_normal, (width, height), interpolation=cv2.INTER_NEAREST)
    equi_confidence = cv2.resize(equi_confidence, (width, height), interpolation=cv2.INTER_NEAREST)

    return equi_image, equi_depth, equi_normal, equi_confidence, equi_mask


def save_outputs(cfg, name, equi_depth, normal_refined):
    out = cfg.save.output_dir
    base_png = name.replace(".jpg", ".png")

    os.makedirs(os.path.join(out, "final_depth"), exist_ok=True)
    os.makedirs(os.path.join(out, "final_normal"), exist_ok=True)

    cv2.imwrite(
        os.path.join(out, "final_depth", base_png),
        (equi_depth[0][0].detach().cpu().numpy() * 1000.0).astype(np.uint16),
    )
    cv2.imwrite(
        os.path.join(out, "final_normal", base_png),
        ((normal_refined * 0.5 + 0.5) * 255).astype(np.uint8),
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = parse_train_config("configs/default_config", args.file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = [
        name for name in os.listdir(os.path.join(cfg.save.output_dir, "image"))
        if name.split(".")[-1].lower().endswith("g")
    ]

    loss_param, opt_param = get_refine_params(cfg)
    total_time = 0.0

    for name in tqdm(names):
        if os.path.exists(os.path.join(cfg.save.output_dir, "final_depth", name.replace(".jpg", ".png"))):
            continue

        equi_image, equi_depth, equi_normal, equi_confidence, equi_mask = load_inputs(cfg, name)

        start = time()
        depth_refined, normal_refined = refine_depth_ERP(
            equi_image,
            equi_depth,
            (cfg.datasets.min_depth, cfg.datasets.max_depth),
            loss_param, opt_param,
            depth_confidence=equi_confidence,
            normal=equi_normal,
            valid_mask=equi_mask,
            device=device,
        )
        elapsed = time() - start
        print(f"refine time: {elapsed:.3f}s")
        total_time += elapsed

        normal_refined = normal_refined.clip(-1, 1)

        if equi_mask is not None:
            depth_refined *= equi_mask
            normal_refined *= equi_mask[..., None]

        equi_depth = cam_height_scale_alignment(
            depth_refined, normal_refined,
            real_height=cfg.datasets.real_height,
            device=device,
            visualize=None,
        )

        save_outputs(cfg, name, equi_depth, normal_refined)

    avg_time = total_time / len(names)
    print(f"Average time: {avg_time:.4f}s")
    with open(os.path.join(cfg.save.output_dir, "avg_time.txt"), "w") as f:
        f.write(f"Average refinement time: {avg_time:.4f} seconds\n")
