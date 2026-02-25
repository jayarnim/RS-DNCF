from .loss.registry import LOSS_FN_REGISTRY


def loss_fn_builder(cfg):
    cls = LOSS_FN_REGISTRY[cfg.name]
    return cls(**cfg.params)