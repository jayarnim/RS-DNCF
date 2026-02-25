import torch
from .selector.registry import SELECTOR_REGISTRY


def selector_builder(
    interactions: torch.Tensor, 
    cfg,
):
    kwargs = dict(
        interactions=interactions,
        **cfg.params,
    )
    cls = SELECTOR_REGISTRY[cfg.name]
    return cls(**kwargs)