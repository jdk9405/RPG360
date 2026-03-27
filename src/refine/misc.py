"""
inspired by https://github.com/rossimattia/depth-refinement-and-normal-estimation/tree/master
"""

"""
DK Contribution for ERP
<similarity_graph>
- similarity graph connected circularly
    - along the width direction, pixels going out of src or trg bounds wrap around circularly
- via valid_mask, only valid pixels are retained; all others are zeroed out
"""


import math
import numpy as np
import torch
from torch.nn import functional as fun
from .filters import diff_filter_bank

from scipy.ndimage import map_coordinates
def resize_map(data, size_new, order=0):
    """It re-sizes the input map.

    It up-samples or down-samples any map (e.g., an image) with one or more channels.

    Args:
        data: map to resize, arranged as an `(H, W)` or `(H, W, C)` array.
        size_new: 2-tuple specifying the new height and width.
        order: order of the spline to be used in the re-sizing.

    Returns:
        The re-sized map, with dimensions `size_new[0], size_new[1]` or `size_new[0], size_new[1], C`. The output
        data type reflects the input one.
    """

    # Check that the data is either 2D or 3D.
    if (data.ndim != 2) & (data.ndim != 3):
        raise ValueError('Input data must be either 2D or 3D.')

    # Input data dimensions.
    height = data.shape[0]
    width = data.shape[1]

    # The target dimensions.
    height_new, width_new = size_new

    # We make the following assumptions:
    # - each pixel in the input data has height `1` and width `1`,
    # - `data[y, x]` is concentrated at the spatial coordinates `(y, x)`.
    # According to the previous two assumptions:
    # - the top left corner of the pixel associated to `data[y, x]` is at spatial coordinates `(y - 0.5, x - 0.5)`,
    # - the bottom right corner of the pixel associated to `data[y, x]` is at spatial coordinates `(y + 0.5, x + 0.5)`,
    # - `data` has its top left corner at the spatial coordinates `(- 0.5, - 0.5)`,
    # - `data` has its bottom right corner at the spatial coordinates `(height - 1 + 0.5, width - 1 + 0.5)`.

    # NOTE:
    # Re-sizing the input data means enlarging the pixel size, not decreasing the data (image or depth) area.
    # After resizing, the top left and bottom right corners of `data` will still be located at spatial coordinates
    # `(- 0.5, - 0.5)` and `(height - 1 + 0.5, width - 1 + 0.5)`, respectively.

    # New pixel dimensions.
    pixel_height_new = float(height) / height_new
    pixel_width_new = float(width) / width_new

    # Compute the coordinates of center of the top left pixel in the re-sized data.
    start_y = - 0.5 + (pixel_height_new / 2.0)
    start_x = - 0.5 + (pixel_width_new / 2.0)

    # Compute the coordinates of the center of the bottom right pixel in the new data.
    end_y = height - 1 + 0.5 - (pixel_height_new / 2.0)
    end_x = width - 1 + 0.5 - (pixel_width_new / 2.0)

    # Compute the new sampling grid.
    y_coord_new, x_coord_new = np.mgrid[start_y:end_y:(height_new * 1j), start_x:end_x:(width_new * 1j)]

    # Organize the sampling grid in a single array.
    points_new = np.stack((y_coord_new.flatten(), x_coord_new.flatten()), axis=1)

    # Re-sample the input depth.
    if data.ndim == 2:

        # Single channel input.

        aux = map_coordinates(data, points_new.T, order=order, mode='nearest')
        data_resized = np.reshape(aux, (height_new, width_new))

    else:

        # Multiple channel input.

        # Number of channels.
        channel_nb = data.shape[2]

        aux = tuple(
            map_coordinates(data[:, :, i], points_new.T, order=order, mode='nearest') for i in range(channel_nb))
        aux = np.stack(aux, axis=1)
        data_resized = np.reshape(aux, (height_new, width_new, channel_nb))

    return data_resized

def similarity_graph(image,
                     window_size=9, patch_size=7,
                     sigma_intensity=0.2, sigma_spatial=3.0,
                     degree_max=15, 
                     valid_mask=None,
                     fg_mask=None,):

    """It builds a similarity graph on the input image.

    Args:
        image: reference image, arranged as a `(1, 1, H, W)` tensor.
        window_size: edge size of the square searching window.
        patch_size: edge size of the square patch used in the similarity computation.
        sigma_intensity: intensity standard deviation for the gaussian similarity weights.
        sigma_spatial: spatial standard deviation for the gaussian similarity weights.
        degree_max: maximum number of neighbors for each node (pixel) in the similarity graph.

    Returns:
        A tuple containing two `(1, degree_max, H, W)` tensors. The entry `(0, k, i, j)` of the first tensor stores the
        similarity weight between the pixels `(i, j)' of the input image and its k-th best neighbor.
        The linear index of k-th best neighbor is stored in the entry `(0, k, i, j)` of the second tensor.
        A pixel `(i, j)` with less than `degree_max` neighbors has the array `(0, :, i, j)` in the first tensor filled
        with zeros. The linear index, in the second tensor, associated to the aforementioned zero weights is the linear
        index of the pixel `(i, j)` itself.
    """

    # Check the input image type
    assert image.is_floating_point(), "The input image must be of type float."

    # Image dimensions.
    channel_nb = image.size(1)
    height = image.size(2)
    width = image.size(3)

    # Organize the channels in the batch dimension.
    image_aux = image
    if channel_nb > 1:
        image_aux = image.transpose(0, 1).contiguous()

    # Create the filters to be used to compute the patch similarity.
    filter_bank = diff_filter_bank(window_size).to(image_aux)       # [80, 1, 9, 9] # diff window: center=-1, one non-center pixel=1, rest=0

    # Compute the padding for the patch similarity computation.
    window_radius = int((window_size - 1) / 2.0)    # 4 = (9 - 1) / 2
    patch_radius = int((patch_size - 1) / 2.0)      # 1 = (3 - 1) / 2
    pad = [window_radius + patch_radius] * 4        # [5, 5, 5, 5]

    # left right top bottom
    pad1 = [window_radius + patch_radius, window_radius + patch_radius, 0, 0]
    pad2 = [0, 0, window_radius + patch_radius, window_radius + patch_radius]
    temp = fun.pad(image_aux, pad1, mode="circular")
    temp = fun.pad(temp, pad2, mode="replicate")

    # Compute the pixel similarity
    pixel_similarity = fun.conv2d(temp, filter_bank).pow(2).sum(dim=0, keepdim=True)

    # Compute the integral image associated to 'similarity'
    pad = (1, 0, 1, 0)      # (pad_left, pad_right, pad_top, pad_bottom)
    integral = fun.pad(pixel_similarity, pad, mode='constant', value=0).cumsum(dim=2).cumsum(dim=3)  # [1, 80, h+3, w+3]
    # `integral` is `(1, window_size * window_size, height + (2 * patch_radius) + 1, width + (2 * patch_radius) + 1)`.

    # Free the memory associated to `pixel_similarity`.
    del pixel_similarity

    # Exploit the integral image to compute the patch similarity in constant time.
    integral_height = integral.size(2)  # h + 3
    integral_width = integral.size(3)   # w + 3
    bottom_right = integral.narrow(2, integral_height - height, height).narrow(3, integral_width - width, width)        # [1, 80, 192, 192]
    bottom_left = integral.narrow(2, integral_height - height, height).narrow(3, 0, width)                              # [1, 80, 192, 192]
    top_right = integral.narrow(2, 0, height).narrow(3, integral_width - width, width)
    top_left = integral.narrow(2, 0, height).narrow(3, 0, width)                            # equivalent to a crop    # ||Q_i - Q_j|| in paper
    patch_similarity = bottom_right.clone().add_(-1.0, bottom_left).add_(-1.0, top_right).add_(top_left)

    # Normalize the patch similarity.
    patch_similarity.div_((- 2.0) * (sigma_intensity ** 2))

    # Free the memory associated to `integral`.
    del integral

    # Define the window grid.
    y_window, x_window = torch.meshgrid(
        [torch.arange(- window_radius, window_radius + 1, dtype=torch.int16, device=image_aux.device),
         torch.arange(- window_radius, window_radius + 1, dtype=torch.int16, device=image_aux.device)])     # [9, 9]

    y_window = y_window.reshape(1, -1)      # [1, 81]
    x_window = x_window.reshape(1, -1)      # [1, 81]

    # Remove the entry `(0, 0)` from the window grid, as `filter_bank` does not contain any filter for this coordinate.
    mask = (y_window == 0) & (x_window == 0)
    y_window = y_window[~mask].reshape(1, -1, 1, 1)     # [1, 80, 1, 1]
    x_window = x_window[~mask].reshape(1, -1, 1, 1)     # [1, 80, 1, 1]

    # Compute the squared spatial distance.
    spatial_weights = x_window.to(patch_similarity).pow_(2) + y_window.to(patch_similarity).pow_(2)     # [1, 80, 1, 1]

    # Normalize the spatial distance.
    spatial_weights.div_((- 2.0) * (sigma_spatial ** 2))

    # Compute the global weights (based on both patch similarity and spatial distance).
    weights = patch_similarity.add_(spatial_weights).exp_()

    # Define the image grid.
    y_source, x_source = torch.meshgrid(
        [torch.arange(height, dtype=torch.int16, device=image_aux.device),
         torch.arange(width, dtype=torch.int16, device=image_aux.device)])      # [h, w]
    y_source = y_source[None, None,]
    x_source = x_source[None, None,]        # [1, 1, h, w]

    # Detect and remove the non valid weights, i.e., those associated to pixel outside the actual image support.
    y_target = torch.zeros_like(y_source)
    x_target = torch.zeros_like(x_source)   # [1, 1, h, w]
    for i in range(weights.size(1)):

        # Compute the neighbouring pixel coordinates.
        torch.add(y_source, y_window.narrow(1, i, 1), out=y_target)             # [1, 1, h, w]
        torch.add(x_source, x_window.narrow(1, i, 1), out=x_target)

        weights.narrow(1, i, 1).mul_(
            (y_target >= 0).to(weights)).mul_(
            (y_target < height).to(weights))                # [1, 80, h, w]

        # cut edges to invalid source pixels in advance using valid_mask
        # also cut edges to invalid adjacent (target) pixels using valid_mask
        if valid_mask is not None:

            y_ = y_target.clone().long()
            x_ = x_target.clone().long()
            y_[y_ < 0] = 0
            y_[y_ > height-1] = height-1
            x_ = x_ % width
            xy_linear = (width * y_ + x_).view(1, 1, -1)        # [1, 1, h*w]
            mask_for_target = torch.gather(torch.from_numpy(valid_mask[None,None]).view(1,1,-1).to(xy_linear), -1, xy_linear).view(1, 1, height, width)

            weights.narrow(1, i, 1).mul_(
                torch.from_numpy(valid_mask[None,None]).to(weights)).mul_(
                    mask_for_target)

        # if fg_mask is not None:
        #     y_ = y_target.clone().long()
        #     x_ = x_target.clone().long()
        #     y_[y_ < 0] = 0
        #     y_[y_ > height-1] = height-1
        #     x_ = x_ % width
        #     xy_linear = (width * y_ + x_).view(1, 1, -1)        # [1, 1, h*w]
        #     bg_mask = 1 - fg_mask
        #     mask_for_target = torch.gather(torch.from_numpy(bg_mask[None, None]).view(1,1,-1).to(xy_linear), -1, xy_linear).view(1, 1, height, width)

        #     weights.narrow(1, i, 1).mul_(torch.from_numpy(bg_mask[None, None]).to(weights)).mul_(mask_for_target)


    # For each pixel, select the `degree_max` neighbours with the largest weights.
    weights_top, indexes = torch.topk(weights, degree_max, dim=1)       # [1, 20, h, w] top-2 neighbors
    # Note that, although the weights associated to non valid neighbours have been set equal to zero, some of these
    # neighbours may still have been selected. This must be taken into account later.

    # Free the memory associated to `weights`.
    del weights

    # Normalize the vector of weights associated to each pixel by its sum.
    weights_top.div_(
        torch.max(weights_top.sum(dim=1, keepdim=True).expand_as(weights_top), weights_top.new_ones(1) * 1e-12))    # [1, 20, h, w]

    # Build the tensor `indexes_linear`.
    index_linear = torch.zeros_like(weights_top, dtype=torch.long)  # [1, 20, h, w]
    for i in range(degree_max):

        # Flatten the spatial dimensions of `indexes`.
        indexes_flattened = indexes.narrow(1, i, 1).view(1, -1, 1, 1)   # [1, h*w, 1, 1]

        # Compute the neighboring pixel coordinates.
        torch.add(
            y_source,
            torch.gather(y_window, 1, indexes_flattened).view(y_source.size()),
            out=y_target)
        torch.add(
            x_source,
            torch.gather(x_window, 1, indexes_flattened).view(x_source.size()),
            out=x_target)

        # The coordinates of the non valid neighbors of a pixel `p` are set equal to the coordinates of `p` itself.
        mask = None
        if (y_target < 0).any() or (y_target >= height).any():
            mask = (y_target < 0) | (y_target >= height)
            y_target[mask] = y_source[mask]
#        if (x_target < 0).any() or (x_target >= width).any():
#            mask = (x_target < 0) | (x_target >= width)
#            x_target[mask] = x_source[mask]

        # circular index
        x_target = x_target % width        # circular index

        # Convert the spatial indexes into linear.
        torch.add(
            x_target.to(index_linear),
            width,
            y_target.to(index_linear),
            out=index_linear.narrow(1, i, 1))   # x_target + width * y_target: linear index scanning top-left to bottom-right

    # Free the memory associated to `y_target`, `x_target`, `mask`.
    del y_target, x_target, mask

    return weights_top, index_linear        # [1, 20, h, w], [1, 20, h, w]


