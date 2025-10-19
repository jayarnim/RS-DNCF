import torch.nn as nn
import torch.optim as optim
from .scheduler.registry import SCHEDULER_REGISTRY


def scheduler_builder(
    optimizer: optim.Optimizer, 
    cfg: dict,
):
    NAME = cfg["name"]
    KWARGS = cfg["args"]
    sch_cls = SCHEDULER_REGISTRY[NAME]
    return sch_cls(optimizer, **KWARGS)