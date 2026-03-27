import os
import sys
import random

import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.camera.Projection import EquirecGrid


OUTPUT_DIR = "./workdir/mp3d"
DEPTH_SCALE = 1000.0
VIS_HEIGHT = 320
VIS_WIDTH = 640

random.seed(12221)
EG = EquirecGrid()


if __name__ == "__main__":
    names = [
        name for name in os.listdir(os.path.join(OUTPUT_DIR, "image"))
        if name.split(".")[-1].lower().endswith("g")
    ]
    random.shuffle(names)

    for name in tqdm(names):
        img_path = os.path.join(OUTPUT_DIR, "image", name)
        depth_path = os.path.join(OUTPUT_DIR, "final_depth", name.replace(".jpg", ".png"))

        equi_image = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        equi_depth = (cv2.imread(depth_path, -1) / DEPTH_SCALE).astype(np.float32)

        equi_image = cv2.resize(equi_image, (VIS_WIDTH, VIS_HEIGHT), interpolation=cv2.INTER_LINEAR)
        equi_depth = cv2.resize(equi_depth, (VIS_WIDTH, VIS_HEIGHT), interpolation=cv2.INTER_NEAREST)

        equi_image = torch.from_numpy(equi_image)                           # [H, W, 3]
        equi_depth = torch.from_numpy(equi_depth).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        xyz = EG.to_xyz(equi_depth)[0].view(3, -1)
        xyz_rgb = equi_image.reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.T)
        pcd.colors = o3d.utility.Vector3dVector(xyz_rgb)

        print(f"Visualizing: {name}")
        o3d.visualization.draw_geometries([pcd])
