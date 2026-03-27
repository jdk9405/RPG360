import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import math
import numpy as np

import open3d as o3d
import random

from src.utils import BackprojectDepth, NormalSurface
from src.camera.Projection import Cube2Equirec, EquirecGrid

import ipdb

def get_intrinsics(cubemap_dim=384, device=torch.device("cpu")):

    f = 0.5 * cubemap_dim / np.tan(0.5 * np.pi / 2.0)
    K = torch.eye(3)[None].repeat(6, 1, 1).to(device)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = cubemap_dim * 0.5
    K[:, 1, 2] = cubemap_dim * 0.5
    # print(K)
    inv_K = K.inverse()

    return K, inv_K


def get_cam_pts(backproject_depth, cube_depth, inv_K):
    cube_cam_pts = backproject_depth(cube_depth, inv_K=inv_K)       # [6, 4, N]
    cube_cam_pts = cube_cam_pts[:, :3, :]                           # [6, 3, N]

    return cube_cam_pts

def get_transformed_cube_normal(cube_cam_pts, cube_normal, cube_R, device=torch.device("cpu")):
    
    # backproject_depth =  BackprojectDepth(batch_size=6, height=384, width=384).to(device)
    surface_normal = NormalSurface(batch_size=6, height=384, width=384).to(device)

    cube_normal_from_depth = surface_normal.get_surface_normal(cube_cam_pts)

    
    # 좌표계 front 로 일치
    transformed_normal_list = []
    transformed_normal_from_depth_list = []
    for R, normal, normal_from_depth in zip(cube_R, cube_normal, cube_normal_from_depth):

        # normal
        transformed_normal = R.inverse() @ normal.reshape(3, -1)        # [3, N]
        transformed_normal = transformed_normal.reshape(3, 384, 384)
        transformed_normal_list.append(transformed_normal)
        # normal from depth
        transformed_normal_from_depth = R.inverse() @ normal_from_depth.reshape(3, -1)
        transformed_normal_from_depth = transformed_normal_from_depth.reshape(3, 384, 384)
        transformed_normal_from_depth_list.append(transformed_normal_from_depth)
    transformed_normal = torch.stack(transformed_normal_list, dim=0)
    transformed_normal_from_depth = torch.stack(transformed_normal_from_depth_list, dim=0)

    return transformed_normal, transformed_normal_from_depth


def cubemap_scale_alignment(cube_depth, cube_cam_pts, cube_normal, real_height=1.0, device=torch.device("cpu")):

    """ down """
    # Estimate scale ratio! find the scale from "down" image
    down_normal = cube_normal[1]            # [3, 384, 384]
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    th_degree = 10
    threshold = math.cos(math.radians(th_degree))
    ones, zeros = torch.ones(1, 384, 384), torch.zeros(1, 384, 384)
    vertical = torch.cat([zeros, -ones, zeros], dim=0).to(device)       # [3, 384, 384]    # 바닥 normal 방향이 (0, -1, 0) 으로 정의됨.

    cosine_sim = cos(down_normal, vertical)
    vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

    down_cam_pts = cube_cam_pts[1]      # [3, 384*384]
    esti_heights = down_cam_pts[2]        # [384*384]
    esti_heights = esti_heights.reshape(384, 384)
    esti_height = esti_heights[vertical_mask].median()

    if real_height is not None:
        height_ratio = real_height / esti_height
    else:
        height_ratio = 1.

    rescaled_cube_depth = cube_depth.clone()
    rescaled_cube_depth[1] = cube_depth[1] * height_ratio
    
    """ back """
    down_z_vals = esti_heights[-1, :] * height_ratio
    back_cam_pts = cube_cam_pts[0].reshape(3, 384, 384)
    back_y_vals = back_cam_pts[1][-1, :]
    reverse_idx = list(reversed(range(len(back_y_vals))))
    back_y_vals = back_y_vals[reverse_idx]
    back_valid = back_y_vals > 0
    down_z_vals, back_y_vals = down_z_vals[back_valid], back_y_vals[back_valid]
    back_down_ratio = (down_z_vals / back_y_vals).median()
    rescaled_cube_depth[0] = cube_depth[0] * back_down_ratio

    """ front """
    down_z_vals = esti_heights[0, :] * height_ratio
    front_cam_pts = cube_cam_pts[2].reshape(3, 384, 384)
    front_y_vals = front_cam_pts[1][-1, :]
    front_valid = front_y_vals > 0
    down_z_vals, front_y_vals = down_z_vals[front_valid], front_y_vals[front_valid]
    front_down_ratio = (down_z_vals / front_y_vals).median()
    rescaled_cube_depth[2] = cube_depth[2] * front_down_ratio

    """ left """
    down_z_vals = esti_heights[:, 0] * height_ratio
    left_cam_pts = cube_cam_pts[3].reshape(3, 384, 384)
    left_y_vals = left_cam_pts[1][-1, :]
    reverse_idx = list(reversed(range(len(left_y_vals))))
    left_y_vals = left_y_vals[reverse_idx]
    left_valid = left_y_vals > 0
    down_z_vals, left_y_vals = down_z_vals[left_valid], left_y_vals[left_valid]
    left_down_ratio = (down_z_vals / left_y_vals).median()
    rescaled_cube_depth[3] = cube_depth[3] * left_down_ratio

    """ right """
    down_z_vals = esti_heights[:, -1] * height_ratio
    right_cam_pts = cube_cam_pts[4].reshape(3, 384, 384)
    right_y_vals = right_cam_pts[1][-1, :]
    right_valid = right_y_vals > 0
    down_z_vals, right_y_vals = down_z_vals[right_valid], right_y_vals[right_valid]
    right_down_ratio = (down_z_vals / right_y_vals).median()
    rescaled_cube_depth[4] = cube_depth[4] * right_down_ratio

    """ top """
    top_cam_pts = cube_cam_pts[5].reshape(3, 384, 384)
    top_z_vals1 = torch.flip(top_cam_pts[2][0,:], dims=[0])
    top_z_vals2 = top_cam_pts[2][:,0]
    top_z_vals3 = top_cam_pts[2][-1,:]
    top_z_vals4 = torch.flip(top_cam_pts[2][:,-1], dims=[0])
    top_z_vals = torch.cat([top_z_vals1, top_z_vals2, top_z_vals3, top_z_vals4], dim=0)
    back_y_vals = -(back_cam_pts*back_down_ratio)[1][0,:]
    left_y_vals = -(left_cam_pts*left_down_ratio)[1][0,:]
    front_y_vals = -(front_cam_pts*front_down_ratio)[1][0,:]
    right_y_vals = -(right_cam_pts*right_down_ratio)[1][0,:]
    horizon_y_vals = torch.cat([back_y_vals, left_y_vals, front_y_vals, right_y_vals], dim=0)
    top_valid = horizon_y_vals > 0.0
    top_horizon_ratio = (horizon_y_vals[top_valid] / top_z_vals[top_valid]).median()
    rescaled_cube_depth[5] = cube_depth[5] * top_horizon_ratio

    return rescaled_cube_depth


def get_confidence_map(cube_normal, cube_normal_from_depth,
                       kernel_size=(5,5), sigma=(0.1, 5),
                       min_conf=0.0):
    # get confidence map
    cos = nn.CosineSimilarity()
    cube_confidence = cos(cube_normal, cube_normal_from_depth).unsqueeze(1)    # [6, 1, 384, 384]

    # smoothing with gaussian kernel
    blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    cube_confidence = blurrer(cube_confidence)
    cube_confidence = torch.clip(cube_confidence, min=min_conf, max=1.0)

    return cube_confidence


def get_ERP_from_cubemap(cube, cube_depth, cube_normal, cube_confidence, cube_mask=None,
                         cube_length=384, equi_h=480,
                         device=torch.device("cpu")):

    # c2e = Cube2Equirec(384, 1920).to(device)
    c2e = Cube2Equirec(cube_length, equi_h).to(device)
    equi_depth = c2e(cube_depth, is_depth=True, mode="nearest")                 # [1, 1, h, w]
    equi_image = c2e(cube)                                      # [1, 3, h, w]
    equi_normal = c2e(cube_normal)                              # [1, 3, h, w]

    EG = EquirecGrid()
    xyz = EG.to_xyz(equi_depth)[0].view(3, -1)

    ones = torch.ones(1, 1, equi_h, equi_h*2).to(equi_depth)
    cam_points = torch.cat([xyz.view(1, 3, equi_h, equi_h*2), ones], dim=1)
    nei=1
    cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
    cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
    cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
    cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
    cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
    cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
    cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
    cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
    cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

    vector_x0   = cam_points_x0   - cam_points_ctr
    vector_y0   = cam_points_y0   - cam_points_ctr
    vector_x1   = cam_points_x1   - cam_points_ctr
    vector_y1   = cam_points_y1   - cam_points_ctr
    vector_x0y0 = cam_points_x0y0 - cam_points_ctr
    vector_x0y1 = cam_points_x0y1 - cam_points_ctr
    vector_x1y0 = cam_points_x1y0 - cam_points_ctr
    vector_x1y1 = cam_points_x1y1 - cam_points_ctr

    # # outward normal
    # normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
    # normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
    # normal_2 = F.normalize(torch.cross(vector_x0y1, vector_x0y0, dim=1), dim=1).unsqueeze(0)
    # normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)
    # inward normal
    normal_0 = F.normalize(torch.cross(vector_y0,   vector_x0,   dim=1), dim=1).unsqueeze(0)
    normal_1 = F.normalize(torch.cross(vector_y1,   vector_x1,   dim=1), dim=1).unsqueeze(0)
    normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
    normal_3 = F.normalize(torch.cross(vector_x1y1, vector_x1y0, dim=1), dim=1).unsqueeze(0)

    normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
    normals = F.normalize(normals, dim=1)

    refl = nn.ReflectionPad2d(nei)
    equi_normal_from_depth = refl(normals)

    equi_confidence = c2e(cube_confidence)                          # [1, 1, 1920, 3840]
    equi_mask = (equi_confidence.clone() > 0.0)[0, 0].detach().cpu().numpy()

    if cube_mask is not None:
        equi_mask = c2e(cube_mask, mode="nearest")


    return equi_image, equi_depth, equi_normal, equi_normal_from_depth, equi_confidence, equi_mask 


def cam_height_scale_alignment(equi_depth, equi_normal,
                               real_height=1.0, th_degree=10,
                               device=torch.device("cpu"),
                               visualize=None, equi_image=None):


    height, width = equi_depth.shape
    equi_depth = torch.from_numpy(equi_depth)[None, None]
    equi_normal = torch.from_numpy(equi_normal).permute(2,0,1)[None]


    EG = EquirecGrid()
    xyz = EG.to_xyz(equi_depth)[0].view(3, -1)

    # ones = torch.ones(1, 1, 480, 960)
    # cam_points = torch.cat([xyz.view(1, 3, 480, 960), ones], dim=1)
    # nei=1
    # cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
    # cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
    # cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
    # cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
    # cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
    # cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
    # cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
    # cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
    # cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

    # vector_x0   = cam_points_x0   - cam_points_ctr
    # vector_y0   = cam_points_y0   - cam_points_ctr
    # vector_x1   = cam_points_x1   - cam_points_ctr
    # vector_y1   = cam_points_y1   - cam_points_ctr
    # vector_x0y0 = cam_points_x0y0 - cam_points_ctr
    # vector_x0y1 = cam_points_x0y1 - cam_points_ctr
    # vector_x1y0 = cam_points_x1y0 - cam_points_ctr
    # vector_x1y1 = cam_points_x1y1 - cam_points_ctr

    # # inward normal
    # normal_0 = F.normalize(torch.cross(vector_y0,   vector_x0,   dim=1), dim=1).unsqueeze(0)
    # normal_1 = F.normalize(torch.cross(vector_y1,   vector_x1,   dim=1), dim=1).unsqueeze(0)
    # normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
    # normal_3 = F.normalize(torch.cross(vector_x1y1, vector_x1y0, dim=1), dim=1).unsqueeze(0)
    # normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
    # normals = F.normalize(normals, dim=1)
    # refl = nn.ReflectionPad2d(nei)
    # normals = refl(normals)
    # normals = normals.squeeze(0)

    threshold = math.cos(math.radians(th_degree))
    ones, zeros = torch.ones(1, height, width), torch.zeros(1, height, width)
    vertical = torch.cat([zeros, -ones, zeros], dim=0).to(device)       # [3, 384, 384]    # 바닥 normal 방향이 (0, -1, 0) 으로 정의됨.

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_sim_final = cos(equi_normal.squeeze(0), vertical.detach().cpu())
#    vertical_mask = (cosine_sim_final > threshold) | (cosine_sim_final < -threshold)
    vertical_mask = (cosine_sim_final > threshold)

    refined_heights = xyz.view(3, height, width)[1][vertical_mask]
    valid_height = refined_heights > 0.0
    refined_heights = refined_heights[valid_height]
    refined_height = refined_heights.median()

    if (vertical_mask.sum().item() > 0) and (real_height is not None):
        refined_height_ratio = real_height / refined_height
        equi_depth = equi_depth * refined_height_ratio.item()
    else:
        pass


    if visualize is not None:

        pcd = o3d.geometry.PointCloud()
        xyz = EG.to_xyz(equi_depth)[0].view(3, -1)
        # xyz_rgb = equi_image[0].view(3, -1).detach().cpu() 
        xyz_rgb = equi_image.reshape(-1, 3)

        pcd.points = o3d.utility.Vector3dVector(xyz.T)
        # pcd.colors = o3d.utility.Vector3dVector(xyz_rgb.T/256.0)
        pcd.colors = o3d.utility.Vector3dVector(xyz_rgb)
        o3d.visualization.draw_geometries([pcd])

    return equi_depth


def visualize_cube_pcds(K, cube_R, cube_cam_pts, cube):

    flat_cube_rgb = cube.permute(0, 2, 3, 1).reshape(6, -1, 3)       # [6, N, 3]

    # 좌표계 front 로 일치
    pcd_list = []
    camera_list = []
    transformed_cube_cam_pts = []
    for intrinsic, R, cam_pts, flat_rgb in zip(K, cube_R, cube_cam_pts, flat_cube_rgb):
        pcd = o3d.geometry.PointCloud()
        transformed_cam_pts = R.inverse() @ cam_pts       # [3, N]
        transformed_cube_cam_pts.append(transformed_cam_pts)

#        mask = transformed_cam_pts[1:2, :] > 0                         # if you want to eliminate the ceiling pointcloud 
#        transformed_cam_pts = transformed_cam_pts * mask
        
        pcd.points = o3d.utility.Vector3dVector(transformed_cam_pts.permute(1,0).detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(flat_rgb.detach().cpu().numpy() / 255.0)
        pcd_list.append(pcd)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.detach().cpu().numpy()
        frustrum = o3d.geometry.LineSet.create_camera_visualization(view_width_px=384, view_height_px=384,
                                                                    intrinsic=intrinsic.detach().cpu().numpy(),
                                                                    extrinsic=extrinsic)
        frustrum.paint_uniform_color([random.random(), random.random(), random.random()])
        camera_list.append(frustrum)

    transformed_cube_cam_pts = torch.stack(transformed_cube_cam_pts, dim=0)       # [6, 3, N]
    # rescaled point cloud visualization
    o3d.visualization.draw_geometries(pcd_list + camera_list)

