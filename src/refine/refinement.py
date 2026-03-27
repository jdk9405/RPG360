"""
inspired by https://github.com/rossimattia/depth-refinement-and-normal-estimation/tree/master
"""

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import device as dev
from cv2 import cvtColor, COLOR_RGB2GRAY

from .losses import Loss
from .misc import resize_map


def get_refine_params(cfg):
    loss_param = [None] * cfg.refine.scale_nb
    opt_param = [None] * cfg.refine.scale_nb
    for i in range(cfg.refine.scale_nb):
        loss_param[i] = {
            'lambda_depth_consistency': cfg.refine.lambda_depth_consistency[i],
            'lambda_normal_consistency': cfg.refine.lambda_normal_consistency[i],
            'lambda_regularization': cfg.refine.lambda_regularization[i],
            'gamma_regularization': cfg.refine.gamma_regularization[i],
            'window_size': cfg.refine.window_size[i],
            'patch_size': cfg.refine.patch_size[i],
            'sigma_intensity': cfg.refine.sigma_int[i],
            'sigma_spatial': cfg.refine.sigma_spa[i],
            'degree_max': cfg.refine.degree_max[i],
            'regularization': cfg.refine.regularization,
        }
        opt_param[i] = {
            'iter_max': cfg.refine.iter_max[i],
            'eps_stop': cfg.refine.eps_stop[i],
            'attempt_max': cfg.refine.attempt_max[i],
            'learning_rate': {
                'lr_start': cfg.refine.lr_start[i],
                'lr_slot_nb': cfg.refine.lr_slot_nb[i],
            },
        }
    return loss_param, opt_param


def refine_depth_ERP(image, depth, depth_range,
                     loss_param, opt_param,
                     depth_confidence=None,
                     normal=None,
                     valid_mask=None,
                     fg_mask=None,
                     depth_init=None, normal_init=None,
                     device=dev('cpu'),
                     debug=False):
    """
    Refines the input depth map and estimates the corresponding normal map in a multi-scale fashion.

    We are inspired by Mattia Rossi, Mireille El Gheche, Andreas Kuhn, Pascal Frossard,
    "Joint Graph-based Depth Refinement and Normal Estimation",
    in IEEE Computer Vision and Pattern Recognition Conference (CVPR), Seattle, WA, USA, 2020.

    The `loss_param` input parameter contains a list of dictionaries, one for each scale. Each dictionary must contain
    the following keys:
    - lambda_depth_consistency: depth consistency term multiplier.
    - lambda_normal_consistency: normal consistency term multiplier.
    - lambda_regularization: depth regularization term multiplier.
    - gamma_regularization: depth regularization term internal multiplier.
    - window_size: search window size (window_size x window_size) to be used in the graph construction.
    - patch_size: patch size (patch_size x patch_size) to be used in the graph construction.
    - sigma_intensity: color difference standard deviation for patch comparison in the graph construction.
    - sigma_spatial: euclidean distance standard deviation for patch comparison in the graph construction.
    - degree_max: maximum number of per pixel neighbors in the graph.
    - regularization: regularization type (0 for NLTGV, 1 for our regularization).

    The `opt_param` input parameter contains a list of dictionaries, one for each scale. Each dictionary must contain
    the following keys:
    - iter_max: maximum number of iterations.
    - eps_stop: minimum relative change between the current and the previous iteration depth maps.
    - attempt_max: maximum number of iterations without improving the loss.
    - learning_rate: dictionary containing the following keys:
        - lr_start: initial learning rate.
        - lr_slot_nb: number of partitions; each partition adopts a learning rate which is 1/10 of those employed at
                      the previous partition; 0 excludes the relative depth map change stopping criterium.
    Args:
        image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
        depth: depth map to refine, arranged as an `(H, W)` array.
        depth_range: depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
        loss_param: list of dictionaries, each one containing the loss parameters for a given scale.
        opt_param: list of dictionaries, each one containing the solver parameters for a given scale.
        depth_confidence: confidence map associated to the depth map to refine. It must have entries in `[0, 1]`.
        normal: 3D normal map to refine, arranged as an `(H, W, 3)` array.
        depth_init: initial guess for the refined depth map.
        normal_init: initial guess for the 3D normal map associated to the refined depth map.
        device: device on which the computation will take place.

    Returns:
        The refined depth map and the corresponding normal map.
    """

    scale_nb = len(opt_param)

    # Build the multi-scale pyramid (coarse to fine, index 0 = finest).
    def _make_pyramid(data, scale_nb, downscale_fn):
        pyramid = [data]
        for _ in range(1, scale_nb):
            prev = pyramid[-1]
            if prev is None:
                pyramid.append(None)
            else:
                size = (int(round(prev.shape[0] / 2.0)), int(round(prev.shape[1] / 2.0)))
                pyramid.append(downscale_fn(prev, size))
        return pyramid

    scale_pyramid = [(image.shape[0], image.shape[1])]
    for i in range(1, scale_nb):
        h = int(round(scale_pyramid[-1][0] / 2.0))
        w = int(round(scale_pyramid[-1][1] / 2.0))
        scale_pyramid.append((h, w))

    image_pyramid = _make_pyramid(image, scale_nb, lambda x, s: resize_map(x, s, order=1))
    depth_pyramid = _make_pyramid(depth, scale_nb, lambda x, s: resize_map(x, s, order=0))
    depth_confidence_pyramid = _make_pyramid(depth_confidence, scale_nb, lambda x, s: resize_map(x, s, order=0))
    normal_pyramid = _make_pyramid(normal, scale_nb, lambda x, s: resize_map(x, s, order=0))
    valid_mask_pyramid = _make_pyramid(valid_mask, scale_nb, lambda x, s: resize_map(x, s, order=0))
    fg_mask_pyramid = _make_pyramid(fg_mask, scale_nb, lambda x, s: resize_map(x, s, order=0))

    # depth_init and normal_init are only needed at the coarsest scale.
    depth_init_pyramid = [None] * scale_nb
    normal_init_pyramid = [None] * scale_nb
    if depth_init is not None:
        depth_init_pyramid[scale_nb - 1] = resize_map(depth_init, scale_pyramid[-1], order=0)
    if normal_init is not None:
        normal_init_pyramid[scale_nb - 1] = resize_map(normal_init, scale_pyramid[-1], order=0)

    # Reverse so index 0 = coarsest scale.
    scale_pyramid.reverse()
    image_pyramid.reverse()
    depth_pyramid.reverse()
    depth_confidence_pyramid.reverse()
    normal_pyramid.reverse()
    depth_init_pyramid.reverse()
    normal_init_pyramid.reverse()
    valid_mask_pyramid.reverse()
    fg_mask_pyramid.reverse()

    distance_scale_params = tuple(
        nn.Parameter(torch.tensor(1.0), requires_grad=True) for _ in range(6)
    )

    depth_refined_pyramid = [None] * scale_nb
    normal_refined_pyramid = [None] * scale_nb

    for i in range(scale_nb):
        if i > 0:
            depth_init_pyramid[i] = resize_map(depth_refined_pyramid[i - 1], scale_pyramid[i], order=0)
            if normal_refined_pyramid[i - 1] is not None:
                normal_init_pyramid[i] = resize_map(normal_refined_pyramid[i - 1], scale_pyramid[i], order=0)

        depth_refined, normal_refined = refine_ERP(
            image_pyramid[i], depth_pyramid[i], depth_range,
            loss_param[i], opt_param[i],
            distance_scale_params=distance_scale_params,
            depth_confidence=depth_confidence_pyramid[i],
            depth_init=depth_init_pyramid[i],
            normal=normal_pyramid[i],
            normal_init=normal_init_pyramid[i],
            valid_mask=valid_mask_pyramid[i],
            fg_mask=fg_mask_pyramid[i],
            device=device,
            debug=debug,
            scale_nb=i,
        )

        depth_refined_pyramid[i] = depth_refined
        normal_refined_pyramid[i] = normal_refined

    return depth_refined_pyramid[-1], normal_refined_pyramid[-1]


def refine_ERP(image, depth, depth_range,
               loss_param, opt_param,
               distance_scale_params,
               depth_confidence=None,
               normal=None,
               depth_init=None,
               normal_init=None,
               valid_mask=None,
               fg_mask=None,
               device=dev('cpu'),
               debug=False,
               scale_nb=None):
    """Implements one scale of the multi-scale pyramid of `refine_depth_ERP`.

    Args:
        image: reference image, arranged as an `(H, W)` or `(H, W, C)` array.
        depth: depth map to refine, arranged as an `(H, W)` array.
        depth_range: depth values must belong to the interval `[depth_range[0], depth_range[1]]`.
        loss_param: dictionary containing the loss parameters.
        opt_param: dictionary containing the solver parameters.
        depth_confidence: confidence map associated to the depth map to refine. It must have entries in `[0, 1]`.
        normal: 3D normal map to refine, arranged as an `(H, W, 3)` array.
        depth_init: initial guess for the refined depth map.
        normal_init: initial guess for the 3D normal map associated to the refined depth map.
        device: device on which the computation will take place.

    Returns:
        The refined depth map and the corresponding normal map.
    """

    height, width = image.shape[:2]
    assert depth.shape == (height, width), \
        'Input depth map size not compatible with the reference image one.'
    if depth_confidence is not None:
        assert depth_confidence.shape == (height, width), \
            'Input depth map confidence size not compatible with the reference image one.'
    if normal is not None:
        assert normal.shape == (height, width, 3), \
            'Input normal map size not compatible with the reference image one.'
    if depth_init is not None:
        assert depth_init.shape == (height, width), \
            'Input initial depth map size not compatible with the reference image one.'
    if normal_init is not None:
        assert normal_init.shape == (height, width, 3), \
            'Input initial normal map size not compatible with the reference image one.'
    if valid_mask is not None:
        assert valid_mask.shape == (height, width), \
            'Input valid mask map size not compatible with the reference image one.'
    if fg_mask is not None:
        assert fg_mask.shape == (height, width), \
            'Input seg mask map size not compatible with the reference image one.'

    if depth.dtype == np.float32:
        depth_dtype = torch.float
    elif depth.dtype == np.float64:
        depth_dtype = torch.double
    else:
        raise TypeError('The input depth map must be either of type double or float.')

    # Convert reference image to grayscale.
    image_gray = image
    if image_gray.ndim == 3:
        # cvtColor requires float32; convert back to original dtype afterwards.
        image_gray = cvtColor(image_gray.astype(np.float32), COLOR_RGB2GRAY).astype(image.dtype)

    loss = Loss(
        image_gray, depth, depth_range,
        loss_param, distance_scale_params,
        valid_mask=valid_mask,
        fg_mask=fg_mask,
        depth_confidence=depth_confidence,
        normal=normal,
        depth_init=depth_init,
        normal_init=normal_init,
        device=device,
    ).to(device=device, dtype=depth_dtype)

    iter_max = opt_param['iter_max']
    learning_rate_start = opt_param['learning_rate']['lr_start']
    learning_rate_slot_nb = opt_param['learning_rate']['lr_slot_nb']

    if learning_rate_slot_nb < 1:
        eps_stop = opt_param['eps_stop']
        attempt_max = opt_param['attempt_max']
        scheduler_step_size = iter_max * 2
    else:
        eps_stop = 0.0
        attempt_max = float('inf')
        scheduler_step_size = int(math.ceil(float(iter_max) / float(learning_rate_slot_nb)))

    loss_history = np.zeros(iter_max + 1)
    optimizer = torch.optim.Adam(loss.parameters(), lr=learning_rate_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, gamma=0.1)

    loss_value_min = float('inf')
    attempt_counter = 0
    relative_depth_change = float('inf')

    # Evaluate initial loss (iteration 0).
    optimizer.zero_grad()
    loss_value, depth_consistency_value, normal_consistency_value, regularization_value = loss.forward()
    with torch.no_grad():
        loss_history[0] = loss_value.item()

    for i in range(1, iter_max + 1):
        loss_value.backward()
        depth_old = loss.depth.clone().detach()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            loss.depth.data = loss.depth.data.clamp(depth_range[0], depth_range[1])

        optimizer.zero_grad()
        loss_value, depth_consistency_value, normal_consistency_value, regularization_value = loss.forward()

        with torch.no_grad():
            relative_depth_change = (
                torch.norm((depth_old - loss.depth).view(-1, 1)) /
                torch.norm(depth_old.view(-1, 1))
            )

            if loss_history[i] >= loss_value_min:
                attempt_counter += 1
            else:
                attempt_counter = 0
                loss_value_min = loss_history[i]

            if (relative_depth_change <= eps_stop) or (attempt_counter >= attempt_max):
                break

        if debug and scale_nb is not None:
            os.makedirs("debug", exist_ok=True)
            depth_dbg = loss.depth.detach().cpu().numpy().squeeze()
            normal_dbg = loss.normal.detach().cpu().numpy().squeeze().transpose(1, 2, 0).clip(-1, 1)
            cv2.imwrite(
                os.path.join("debug", f"depth_{scale_nb:02d}_{i:04d}.png"),
                (depth_dbg * 1000.0).astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join("debug", f"normal_{scale_nb:02d}_{i:04d}.png"),
                ((normal_dbg * 0.5 + 0.5) * 255).astype(np.uint8),
            )

    depth_refined = loss.depth.detach().cpu().numpy().squeeze()
    normal_refined = loss.normal.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

    return depth_refined, normal_refined
