"""
inspired by https://github.com/rossimattia/depth-refinement-and-normal-estimation/tree/master
"""

import ipdb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from .misc import similarity_graph
import numpy as np

from src.camera.Projection import Cube2Equirec, EquirecGrid
from src.camera.Conversion import EquirecTransformer


class Loss(nn.Module):
    def __init__(self, image, depth, depth_range,
                 loss_param, distance_scale_params,
                 valid_mask=None, fg_mask=None,
                 depth_confidence=None, normal=None,
                 depth_init=None, normal_init=None,
                 device=torch.device("cpu")):
        super(Loss, self).__init__()


        if depth_range[0] <= 0 or depth_range[1] == float('inf') or depth_range[0] > depth_range[1]:
            raise ValueError('Invalid depth range.')
        self.depth_min = depth_range[0]
        self.depth_max = depth_range[1]

        # Register the optimization variable
        if depth_init is not None:
            aux = torch.as_tensor(depth_init[None, None, ])
        else:
            aux = torch.as_tensor(depth[None, None, ])     # [1, 1, h_, w_]
        self.depth = nn.Parameter(aux.clone(), requires_grad=True)

        if normal_init is not None:
            aux = torch.as_tensor((np.transpose(normal_init, (2, 0, 1))[None, ]).copy())
        else:
            aux = torch.as_tensor((np.transpose(normal, (2, 0, 1))[None, ]).copy())        # [1, 2, h_, w_]
        self.normal = nn.Parameter(aux.clone(), requires_grad=True)

        # Create the 3D normal consistency loss.
        if loss_param['lambda_normal_consistency'] > 0:
            assert normal is not None, 'Cannot activate the normal consistency term with no input normal map.'
            self.normal_consistency_loss = NormalConsistency_DK(
                normal,
                normal_confidence=depth_confidence,
                valid_mask=valid_mask,
                multiplier=loss_param['lambda_normal_consistency'],
                # mode="cos"
                )
        else:
            self.normal_consistency_loss = None

        # Create the planar regularization loss.
        self.regularization_loss = PieceWisePlanarRegularization_3D_ERP(
            image, depth,
            loss_param['gamma_regularization'],
            window_size=loss_param['window_size'],
            patch_size=loss_param['patch_size'],
            sigma_intensity=loss_param['sigma_intensity'],
            sigma_spatial=loss_param['sigma_spatial'],
            degree_max=loss_param['degree_max'],
            version=loss_param['regularization'],
            multiplier=loss_param['lambda_regularization'],
            valid_mask=valid_mask,
            fg_mask=fg_mask,
            device=device,
            )

        # Create the depth consistency loss.
        self.depth_consistency_loss_dk = DistanceConsistencyL1_scale_ERP(
            depth, 
            # down_depth, 
            depth_range,
            valid_mask=valid_mask,
            depth_confidence=depth_confidence,
            multiplier=loss_param["lambda_depth_consistency"],
            )
        # Get the registered scale params for perspective images in cubemap.
        self.s0, self.s1, self.s2, self.s3, self.s4, self.s5 = distance_scale_params


    def forward(self):

        # depth consistency loss
        depth_consistency_loss = self.depth_consistency_loss_dk(self.depth, self.s0, self.s1, self.s2, self.s3, self.s4, self.s5)

        # normal consistency loss
        if self.normal_consistency_loss is not None:
            normal_consistency_loss = self.normal_consistency_loss(self.normal)
        else:
            normal_consistency_loss = self.depth.new_zeros(1, requires_grad=True)

        # regularization loss
        regularization_loss = self.regularization_loss(self.depth, self.normal)

        # Assemble the full loss.
        loss = depth_consistency_loss + normal_consistency_loss + regularization_loss

        return loss, depth_consistency_loss.item(), normal_consistency_loss.item(), regularization_loss.item()


class DistanceConsistencyL1_scale_ERP(nn.Module):
    def __init__(self, depth, depth_range,
                 valid_mask=None, depth_confidence=None,
                 multiplier=0.0):
        super(DistanceConsistencyL1_scale_ERP, self).__init__()

        # Check the input depth range.
        depth_min, depth_max = depth_range
        assert depth_min < depth_max, "The specifided depth range is empty."
        # Extract the depth map confidence
        if depth_confidence is not None:
            assert (depth_confidence >= 0).all() and (depth_confidence <= 1).all(), \
                "Depth confidence entries must belong to [0, 1]."
            confidence = depth_confidence
        else:
            confidence = 1
        # The confidence is set to zero at non valid depth entries.
        confidence = confidence * ((depth > depth_min) & (depth < depth_max))
        # The confidence is set to zero for unvalid regions.
        if valid_mask is not None:
            confidence = confidence * valid_mask

        # regitser confidence map
        self.register_buffer('confidence', torch.as_tensor(confidence[None, None,]))
        # register the depth map
        self.register_buffer('depth', torch.as_tensor(depth[None, None,]))
        # register the normalization constant
        self.norm_const = self.confidence.sum()
        # register the loss multiplier
        self.multiplier = multiplier

        height, width = depth.shape
        self.c2e = Cube2Equirec(384, height).to(self.depth)

        if valid_mask is not None: 
            self.register_buffer("valid_mask", torch.from_numpy(valid_mask[None, None]))
        else:
            self.register_buffer("valid_mask", torch.ones_like(self.confidence))


    def forward(self, depth, 
                s0, s1, s2, s3, s4, s5):

        # Allocate a zero loss in the case that the loss is disabled, i.e., `self.multiplier` is zero.
        loss = depth.new_zeros(1, requires_grad=True)

        scale_map0 = s0[None, None, None].repeat(1, 1, 384, 384)
        scale_map1 = s1[None, None, None].repeat(1, 1, 384, 384)
        scale_map2 = s2[None, None, None].repeat(1, 1, 384, 384)
        scale_map3 = s3[None, None, None].repeat(1, 1, 384, 384)
        scale_map4 = s4[None, None, None].repeat(1, 1, 384, 384)
        scale_map5 = s5[None, None, None].repeat(1, 1, 384, 384)
        scale_map = torch.cat([scale_map0, scale_map1, scale_map2, scale_map3, scale_map4, scale_map5], dim=0)
        scale_map = self.c2e(scale_map)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:

            # Evaluate the loss.
            # depth: updated param, self.depth: prior
            loss = (depth - self.depth * scale_map).mul(self.confidence).mul(self.valid_mask).abs().sum().div(self.norm_const)
            
            # Weight the loss
            loss = self.multiplier * loss

        # print("##### {}  {}  {}  {}  {}  {}".format(s0, s1, s2, s3, s4, s5))

        return loss


class NormalConsistency_DK(nn.Module):
    def __init__(self, normal, normal_confidence, valid_mask, multiplier, mode="l1"):

        super(NormalConsistency_DK, self).__init__()

        # Extract the normal map confidence.
        if normal_confidence is not None:
            assert (normal_confidence >= 0).all() and (normal_confidence <= 1).all(), \
                'Depth confidence entries must belong to [0, 1].'
            confidence = normal_confidence
        else:
            confidence = 1

        # The confidence is set to zero at non valid normal entries
        aux = np.sum(normal ** 2, axis=2)
        confidence = confidence * ((aux > 0) & (aux < float('inf')))

        # register confidence
        self.register_buffer('confidence', torch.as_tensor(confidence[None, None,]))
        # register normal map
        self.register_buffer('normal', torch.as_tensor((np.transpose(normal, (2, 0, 1))[None, ]).copy()))
        # register the normalization constant
        self.norm_const = self.confidence.sum()
        # register the loss multiplier
        self.multiplier = multiplier

        assert mode in ["l1", "cos"]
        self.mode = mode

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        if valid_mask is not None:
            self.register_buffer('valid_mask', torch.from_numpy(valid_mask[None, None]))
        else:
            self.register_buffer('valid_mask', torch.ones_like(self.confidence))

    def forward(self, normal):
        # Allocate a zero loss in the case that loss is disabled, i.e., 'self.multiplier' is zero.
        loss = normal.new_zeros(1, requires_grad=True)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:
            # Evaluate the loss
            if self.mode == "l1":
                loss = (normal - self.normal).mul(self.confidence).mul(self.valid_mask).abs().sum().div(self.norm_const)
            elif self.mode == "cos":
                loss = (1 - self.cos(normal, self.normal))[:,None].mul(self.confidence).mul(self.valid_mask).sum().div(self.norm_const)
            else:
                raise AssertionError

            loss = self.multiplier * loss

        return loss


class PieceWisePlanarRegularization_3D_ERP(nn.Module):
    """This class implements a regularizer promoting piece-wise planar functions.
    """

    def __init__(self,
                 image: np.array, 
                 depth: np.array,
                 gamma: float,
                 window_size: int = 9, patch_size: int = 7,
                 sigma_intensity: float = 0.2, sigma_spatial: float = 3.0,
                 degree_max: int = 15,
                 version: int = 1,
                 multiplier: float = 0.0,
                 valid_mask: np.array = None,
                 fg_mask: np.array = None,
                 device: torch.device = torch.device('cpu'),
                 ):

        super(PieceWisePlanarRegularization_3D_ERP, self).__init__()
        if image.ndim == 2:
            image_aux = torch.as_tensor(image[None, None, ])
        elif image.ndim == 3:
            image_aux = torch.as_tensor((np.transpose(image, (2, 0, 1))[None, ]).copy())
        else:
            raise ValueError('The input image must be either gray scale or RGB.')
        depth_aux = torch.as_tensor(depth[None, None])
        # Image dimensions.
        height = image_aux.size(2)
        width = image_aux.size(3)
        weights, neighbours = similarity_graph(
            image_aux.to(device),
            window_size=window_size, patch_size=patch_size,                     # 9, 3
            sigma_intensity=sigma_intensity, sigma_spatial=sigma_spatial,       # 0.07, 3.0
            degree_max=degree_max,              # 20
            valid_mask=valid_mask,
            fg_mask=fg_mask,
            ) 


        # weights = weights.to('cpu')
        # neighbours = neighbours.to('cpu')
        self.neighbour_nb = weights.size(1)
        # Flatten the spatial dimensions of `weights` and `neighbours`, and register them.
        weights = weights.view(self.neighbour_nb, -1)           # [20, h*w]
        neighbours = neighbours.view(self.neighbour_nb, -1)     # [20, h*w]
        self.register_buffer('weights', weights)
        self.register_buffer('neighbours', neighbours)

        # number of pixels
        pixel_nb = height * width
        # register the normalization constant
        self.norm_const = pixel_nb
        # register the multiplier associated to the surface normal smoothing
        self.gamma = gamma
        # register the regularization type
        if version == 1:
            self.forward_internal = self.ours
        else:
            raise NotImplementedError("The required regualarization does not exists.")
        # register the loss multiplier
        self.multiplier = multiplier
        # register the valid mask
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(1, -1).repeat(self.neighbour_nb, axis=0)
            self.register_buffer("valid_mask", torch.tensor(valid_mask))
        else:
            self.valid_mask = None

        """ ERP """
        self.EG = EquirecGrid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def get_cam_pts(self, depth):
        cam_points = self.EG.to_xyz(depth)      # [1, 3, h, w]
        
        return cam_points
 

    def forward(self, sig1, sig2):

        return self.forward_internal(sig1, sig2)


    def ours(self, sig1, sig2):
        """
        sig1: depth     [1, 1, h, w]
        sig2: normal    [1, 3, h, w]
        """

        # Allocate a zero loss in the case that the loss is disabled, i.e., 'self.multiplier' is zero.
        loss = sig1.new_zeros(1, requires_grad=True)

        # If the loss is enabled, evaluate it.
        if self.multiplier > 0:
            sig1 = self.get_cam_pts(sig1)       # [1, 3, h, w]

            # Expand and flatten `sig1` and `sig2`.
            sig1_flattened = sig1[:, None, ]            # [1, 1, 3, h, w]
            sig1_flattened = sig1_flattened.expand(
                -1, self.neighbour_nb, -1, -1, -1).view(self.neighbour_nb, 3, -1)   # [20, 3, h*w] # self.neighbour_nb=20
            sig2_flattened = sig2[:, None, ]            # [1, 1, 3, h, w]
            sig2_flattened = sig2_flattened.expand(
                -1, self.neighbour_nb, -1, -1, -1).view(self.neighbour_nb, 3, -1)   # [20, 3, h*w]


            # Compute the planar loss for 'difference vector between 3D points on same surface is perpendicular to it's normal vector.'
            neighbour_sig1_flattened = torch.gather(sig1_flattened, 2, self.neighbours[:, None,].expand(-1,3,-1))
            neighbour_sig2_flattened = torch.gather(sig2_flattened, 2, self.neighbours[:, None,].expand(-1,3,-1))
            aux1 = ((sig1_flattened * sig2_flattened).sum(dim=1) - (neighbour_sig1_flattened * sig2_flattened).sum(dim=1))

            # Compute the planar loss for 'neighbour pixels has similar normal vector.'
            aux2 = (sig2_flattened - neighbour_sig2_flattened).norm(dim=1)  # [20 ,2, hw] -> [20, hw]

            if self.valid_mask is not None:
                aux1 = (aux1 * self.weights * self.valid_mask).norm(dim=0).sum()
            else:
                aux1 = (aux1 * self.weights).norm(dim=0).sum()
            if self.valid_mask is not None:
                aux2 = (aux2 * self.weights * self.valid_mask).sum()
            else:
                aux2 = (aux2 * self.weights).sum()

            # Add the contribution of the two losses for planar regularization
            loss = aux1 + (self.gamma * aux2)

            # Normalize the loss
            loss = loss.div(self.norm_const)

            # Weight the loss
            loss = self.multiplier * loss

        return loss



 
 
