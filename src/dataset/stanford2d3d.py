import cv2
import numpy as np
import random
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
import os

import matplotlib.pyplot as plt
import ipdb


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Stanford2D3D(data.Dataset):

    def __init__(self, root_dir, list_file, height, width, max_depth_meters=10.0):
        """
        Args:
            root_dir (string)   : Directory of the Matterport3D_2K dataset.
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
        rgb = Image.open(rgb_name).convert("RGB")
        rgb = cv2.resize(np.array(rgb), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # depth
        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float32) / 512
        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1

        # mask
        mask = torch.ones([self.h, self.w])
        mask[0:int(self.h*0.15), :] = 0
        mask[self.h-int(self.h*0.15):self.h, :] = 0

        inputs["rgb_name"] = rgb_name
        inputs["depth_name"] = depth_name
        inputs["rgb"] = torch.from_numpy(rgb.transpose(2,0,1)).clone().type(torch.float32)
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))
        inputs["val_mask"] = inputs["val_mask"] * mask


        return  inputs

 
if __name__ == "__main__":

    stanford2d3d = Stanford2D3D(root_dir="/media/dongki/d4T/dongki/stanford2d3d/",
                                list_file="/home/dongki/workspace/perception/DANCE/splits/stfd_test.txt",
                                height=512, width=1024)

    for inputs in stanford2d3d:
        # ipdb.set_trace()
        plt.figure()
        plt.imshow(inputs['rgb'].permute(1,2,0)/255.0)
        plt.figure()
        plt.imshow(inputs['gt_depth'][0])
        plt.figure()
        plt.imshow(inputs['val_mask'][0])
        plt.show()
        print(inputs)
