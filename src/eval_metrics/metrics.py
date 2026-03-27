from functools import wraps

from src.camera.Projection import EquirecGrid
from src.eval_metrics.chamfer_distance import ChamferDistance

import sklearn.neighbors as skln

import os
import numpy as np
import torch
import ipdb

EG = EquirecGrid()


# # HELPERS
# # -----------------------------------------------------------------------------
# def to_float(fn):
#     """Helper to convert all metrics into floats."""
#     @wraps(fn)
#     def wrapper(*a, **kw):
#         return {k: float(v) for k, v in fn(*a, **kw).items()}
#     return wrapper
# # -----------------------------------------------------------------------------


"""
Reference
monodepth_benchmark
"""
# POINTCLOUD
# -----------------------------------------------------------------------------
def _metrics_pointcloud(pred, target, th: float):
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).float().mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).float().mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3): return P, P  # No points are correct.

    f = 2*P*R / (P + R)
    iou = P*R / (P + R - (P*R))
    return f, iou


# @to_float
def compute_depth_metrics(gt, pred, mask=None, median_align=True, include_3d_metric=True):
    """
    Computation of metrics between predicted and ground truth depths
    """

    if mask is None:
        mask = gt > 0

    gt[gt<=0.1] = 0.1
    # pred[pred<=0.1] = 0.1
    gt[gt >= 10] = 10
    # pred[pred >= 10] = 10

    gt_depth = gt[mask.int() == 1]
    pred_depth = pred[mask.int() == 1]

    if median_align:
        median = torch.median(gt_depth) / torch.median(pred_depth)
        pred_depth *= median
        pred[pred<=0.1*median] = 0.1*median
        pred[pred>=10*median] = 10*median

    ###########STEP 1: compute delta#######################
    thresh = torch.max((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    ##########STEP 2:compute mean error###################

    rmse = (gt_depth - pred_depth) ** 2 #standard RMSE function
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt_depth) - torch.log10(pred_depth)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_ = torch.mean(torch.abs(gt_depth - pred_depth))

    abs_rel = torch.mean(torch.abs(gt_depth - pred_depth) / gt_depth)

    sq_rel = torch.mean((gt_depth - pred_depth) ** 2 / gt_depth)

    log10 = torch.mean(torch.abs(torch.log10(pred_depth / gt_depth)))

    #SliceNet&OmniDepth
    mae = torch.mean(torch.abs((pred_depth - gt_depth)) / gt_depth)
    
    mre = torch.mean(((pred_depth - gt_depth)** 2) / gt_depth)

    # #New metrics
    # ctu = continuity(pred_depth, gt_depth)
    # p_rmse = pole_RMSE(pred_depth, gt_depth)

    #### 3D metrics
    chamfer = 0.
    f_score = 0.
    iou = 0.
    f_score_20 = 0.
    iou_20 = 0.
    if include_3d_metric:

        # gt = torch.nn.functional.interpolate(gt, size=(100, 200), mode='nearest', align_corners=False)
        # pred = torch.nn.functional.interpolate(pred, size=(100, 200), mode='nearest', align_corners=False)

        if median_align:
            pred *= median

        mask_xyz = mask[0,0].view(-1)                                               # [hw']
        gt_xyz = EG.to_xyz(gt).view(1, 3, -1)[:, :, mask_xyz.int()==1].float().cuda()              # [1, 3, hw']
        pred_xyz = EG.to_xyz(pred).view(1, 3, -1)[:, :, mask_xyz.int()==1].float().cuda()          # [1, 3, hw']

        # downsample_density = 0.2
        # nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=downsample_density, algorithm='kd_tree', n_jobs=-1)
        # gt_pcd = gt_xyz[0].T                    # [hw', 3]
        # pred_pcd = pred_xyz[0].T                # [hw', 3]
        # nn_engine.fit(gt_pcd)
        # rnn_idxs = nn_engine.radius_neighbors(gt_pcd, radius=downsample_density, return_distance=False)
        # mask = np.ones(gt_pcd.shape[0], dtype=np.bool_)
        # for curr, idxs in enumerate(rnn_idxs):
        #     if mask[curr]:
        #         mask[idxs] = 0
        #         mask[curr] = 1
        # gt_down = gt_pcd[mask].float()
        # pred_down = pred_pcd[mask].float()

        pred_nn, gt_nn = ChamferDistance()(pred_xyz.permute(0, 2, 1), gt_xyz.permute(0, 2, 1))
        # pred_nn, gt_nn = ChamferDistance()(pred_down[None].cuda(), gt_down[None].cuda())
        pred_nn, gt_nn = pred_nn.sqrt(), gt_nn.sqrt()

        f1, iou1 = _metrics_pointcloud(pred_nn, gt_nn, th=0.1)
        f2, iou2 = _metrics_pointcloud(pred_nn, gt_nn, th=0.2)

        chamfer = pred_nn.mean() + gt_nn.mean()
        f_score = 100 * f1
        iou = 100 * iou1
        f_score_20 = 100 * f2
        iou_20 = 100 * iou2

    return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3, \
            chamfer, f_score, iou, f_score_20, iou_20


    # return ctu, p_rmse, mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3
    # return mre, mae, abs_, abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):

    def __init__(self, median_align=True, include_3d_metric=True, crop=0):

        self.median_align = median_align
        self.include_3d_metric = include_3d_metric
        self.crop = crop
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/ctu"] = AverageMeter()
        self.metrics["err/p_rms"] = AverageMeter()
        self.metrics["err/mre"] = AverageMeter()
        self.metrics["err/mae"] = AverageMeter()
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()
        # 3D metrics
        self.metrics["3d/chamfer"] = AverageMeter()
        self.metrics["3d/f-score"] = AverageMeter()
        self.metrics["3d/iou"] = AverageMeter()
        self.metrics["3d/f-score-20"] = AverageMeter()
        self.metrics["3d/iou-20"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/ctu"].reset()
        self.metrics["err/p_rms"].reset()
        self.metrics["err/mre"].reset()
        self.metrics["err/mae"].reset()
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()
        # 3D metrics
        self.metrics["3d/chamfer"].reset()
        self.metrics["3d/f-score"].reset()
        self.metrics["3d/iou"].reset()
        self.metrics["3d/f-score-20"].reset()
        self.metrics["3d/iou-20"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, mask=None):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]

        # ctu, p_rms, mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
        #     compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)
        # mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3 = \
        #     compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align)

        if self.crop > 0:
            gt_depth = gt_depth[..., self.crop:-self.crop, :]
            pred_depth = pred_depth[..., self.crop:-self.crop, :]
            mask = mask[..., self.crop:-self.crop, :]

        mre, mae, abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3, chamfer, f_score, iou, f_score_20, iou_20 = \
            compute_depth_metrics(gt_depth, pred_depth, mask, self.median_align, self.include_3d_metric)


        # self.metrics["err/ctu"].update(ctu, N)
        # self.metrics["err/p_rms"].update(p_rms, N)
        self.metrics["err/mre"].update(mre, N)
        self.metrics["err/mae"].update(mae, N)
        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)

        # 3D metrics
        self.metrics["3d/chamfer"].update(chamfer, N)
        self.metrics["3d/f-score"].update(f_score, N)
        self.metrics["3d/iou"].update(iou, N)
        self.metrics["3d/f-score-20"].update(f_score_20, N)
        self.metrics["3d/iou-20"].update(iou_20, N)

    def print(self, dir=None):
        avg_metrics = []
        # avg_metrics.append(self.metrics["err/ctu"].avg)
        # avg_metrics.append(self.metrics["err/p_rms"].avg)
        avg_metrics.append(self.metrics["err/mre"].avg)
        avg_metrics.append(self.metrics["err/mae"].avg)
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        # 3D metrics
        avg_metrics.append(self.metrics["3d/chamfer"].avg)
        avg_metrics.append(self.metrics["3d/f-score"].avg)
        avg_metrics.append(self.metrics["3d/iou"].avg)
        avg_metrics.append(self.metrics["3d/f-score-20"].avg)
        avg_metrics.append(self.metrics["3d/iou-20"].avg)

        # print("\n  "+ ("{:>9} | " * 13).format("ctu", "p_rms", "mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        # print(("&  {: 8.5f} " * 13).format(*avg_metrics))
        # print("\n  "+ ("{:>9} | " * 11).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        # print(("&  {: 8.5f} " * 11).format(*avg_metrics))
        print("\n  "+ ("{:>9} | " * 16).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3", "chamfer", "f-score", "iou", "f-score-20", "iou-20"))
        print(("&  {: 8.5f} " * 16).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result_3d.txt")
            # file = os.path.join(dir, "result_3d_init.txt")
            with open(file, 'w') as f:
                # print("\n  " + ("{:>9} | " * 13).format("ctu", "p_rms", "mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                #                                       "log10", "a1", "a2", "a3"), file=f)
                # print(("&  {: 8.5f} " * 13).format(*avg_metrics), file=f)
                print("\n  " + ("{:>9} | " * 16).format("mre", "mae", "abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3", "chamfer", "f-score", "iou", "f-score-20", "iou-20"), file=f)
                print(("&  {: 8.5f} " * 16).format(*avg_metrics), file=f)




if __name__ == "__main__":

    gt_depth = torch.rand(1, 1, 512, 1024) * 10.
    pred_depth = torch.rand(1, 1, 512, 1024) * 10.
    # gt_depth = torch.rand(1, 1, 100, 200) * 10.
    # pred_depth = torch.rand(1, 1, 100, 200) * 10.
    aa = compute_depth_metrics(gt_depth, pred_depth)
