"""
inspired by https://github.com/fuenwang/BiFusev2
"""

import os
import sys
import cv2
import time
from imageio import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Equirec2Cube(nn.Module):
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([
            [0, -180.0, 0],
            [90.0, 0, 0],
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [-90, 0, 0]
        ], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        grids = self._getCubeGrid()
        
        for i, grid in enumerate(grids):
            self.register_buffer('grid_%d'%i, grid)

        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        self.per_grid = torch.from_numpy(xyz @ np.linalg.inv(K).T)[None]

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        #self.grids = []
        grids = []
        for _, R in enumerate(self.R_lst):
            tmp = xyz @ R # Don't need to transpose since we are doing it for points not for camera
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            grids.append(torch.FloatTensor(lonlat[None, ...]))
        
        return grids
    
    def forward(self, batch, mode='bilinear', is_depth=False):
        [_, _, h, w] = batch.shape
        assert h == self.equ_h and w == self.equ_w
        assert mode in ['nearest', 'bilinear']
        if is_depth: assert mode == "nearest", "Depth should be interpolated with nearest sampling"

        out = []
        for i in range(6):
            grid = getattr(self, 'grid_%d'%i)
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)

            if is_depth:
                """
                ERP to Cube Depth Scale Compensation
                because of tangential plane approximation for spherical plane
                """
                ratio = self.per_grid.norm(p=2, dim=-1)[:, None]
                sample = sample / ratio

            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i::batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out




if __name__ == '__main__':
    
    import ipdb
    import matplotlib.pyplot as plt
    img = imread("IMAGE PATH") / 255.0
    img = cv2.resize(img, (3840, 1920), interpolation=cv2.INTER_LINEAR)
    batch = torch.FloatTensor(img.transpose(2, 0, 1)[None, ...])
    e2c = Equirec2Cube(384, 1920)

    cube = e2c(batch).cpu().numpy()
    face_name = ['back', 'down', 'front', 'left', 'right', 'top']
    for i in range(6):
        face = cube[i, ...].transpose(1, 2, 0)
        plt.subplot(2, 3, i+1)
        plt.title(face_name[i])
        plt.imshow(face)
        
        save_img = face[..., ::-1]
        cv2.imwrite("example_{}.png".format(face_name[i]), save_img * 255.0)
    plt.show()

