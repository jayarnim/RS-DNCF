import torch.nn as nn
from .optimizer.registry import OPTIMIZER_REGISTRY


def optimizer_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        params=model.parameters(), 
        **cfg.params,
    )
    cls = OPTIMIZER_REGISTRY[cfg.name]
    return cls(**kwargs)