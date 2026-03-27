from yacs.config import CfgNode as CN

cfg = CN()


## Datasets
cfg.datasets = CN()
cfg.datasets.root_dir = ""
cfg.datasets.split = ""
cfg.datasets.min_depth = 0.01
cfg.datasets.max_depth = 5.0
cfg.datasets.min_conf = 0.35
cfg.datasets.real_height = 1.0


## Model
cfg.depth_model = CN()
cfg.depth_model.name = ""


## Refine
cfg.refine = CN()
cfg.refine.scale_nb = 1
cfg.refine.lambda_depth_consistency = [0.5]
cfg.refine.lambda_normal_consistency = [10.0]
cfg.refine.lambda_regularization = [50]
cfg.refine.gamma_regularization = [0.5]
cfg.refine.window_size = [9]
cfg.refine.patch_size = [3]
cfg.refine.sigma_int = [0.07]
cfg.refine.sigma_spa = [3.0]
cfg.refine.degree_max = [20]
cfg.refine.regularization = 1                      # 0 for NLTGV, 1 for joint-graph
cfg.refine.iter_max = [300]
cfg.refine.eps_stop = [0.000001]
cfg.refine.attempt_max = [50]
cfg.refine.lr_start = [0.5]
cfg.refine.lr_slot_nb = [2]


## Save
cfg.save = CN()
cfg.save.output_dir = ""


## Internal: populated automatically by config.py, do not set manually
cfg.config = ''
cfg.default = ''


def get_cfg_defaults():
    return cfg.clone()
