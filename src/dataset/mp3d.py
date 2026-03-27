
from __future__ import print_function
import os
import cv2
from PIL import Image
import numpy as np
from struct import unpack
from pyquaternion import Quaternion
import random

import torch
from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt
import ipdb

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split())
    return rgb_depth_list


class Matterport3D(data.Dataset):
    
    def __init__(self, root_dir, list_file, height, width, max_depth_meters=10.0):

        """
        Args:
            root_dir (string)   : Directory of the Matterport3D dataset.
            list_file (string)  : Path to the txt file containing the list of image and depth.
            height, width       : input_size
            max_depth_meters    : maximum depth value
        """

        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.w = width
        self.h = height
        self.max_depth_meters = max_depth_meters

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        
        inputs = {}

        # rgb
        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = Image.open(rgb_name)       # 0 ~ 255
        rgb = cv2.resize(np.array(rgb), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # depth
        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1) / 4000.
        gt_depth = cv2.resize(np.array(gt_depth), (self.w, self.h), interpolation=cv2.INTER_NEAREST)


        inputs["rgb_name"] = rgb_name
        inputs["depth_name"] = depth_name
        inputs["rgb"] = torch.from_numpy(rgb.transpose(2,0,1)).clone().type(torch.float32)
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters) 
                                                       & ~torch.isnan(inputs["gt_depth"]))

        return inputs



if __name__ == "__main__":

    import open3d as o3d
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.camera.Projection import EquirecGrid

    mp3d = Matterport3D(root_dir="/media/dongki/d4T/dongki/mp3d_dk/",
                        list_file="/home/dongki/workspace/perception/DANCE/splits/mp3d_test.txt",
                        height=256, width=512)
                        # height=1024, width=2048)

    for idx, inputs in enumerate(mp3d):

        print(inputs)
        ipdb.set_trace()

