from .defaults import _C as cfg
from yacs.config import CfgNode


def config_to_dict(cfg):
    local_dict = {}
    for key, value in cfg.items():
        if type(value) != CfgNode:
            local_dict[key] = value
        else:
            local_dict[key] = config_to_dict(value)
    return local_dict

cfg.START_EPOCH = 0
cfg.BEST_PRECISION = 0.0
