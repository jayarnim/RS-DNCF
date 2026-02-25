import torch.nn as nn
from .optimizer.registry import OPTIMIZER_REGISTRY


def _base_optimizer(model, cfg):
    NAME = cfg["name"]
    LEARNING_RATE = cfg["lr"]
    WEIGHT_DECAY = cfg["weight_decay"]

    cls = OPTIMIZER_REGISTRY[NAME]

    kwargs = dict(
        params=model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
    )
    return cls(**kwargs)


def _grouped_optimizer(model, cfg):
    NAME = cfg["name"]
    BASE_LEARNING_RATE = cfg["lr"]
    BASE_WEIGHT_DECAY = cfg["weight_decay"]

    args = []

    param_groups = model.param_groups()

    assigned_params = set()

    for name, params in param_groups.items():
        group_cfg = cfg["param_groups"][name].copy()

        LR_SCALE = group_cfg.pop("lr_scale", 1.0)
        WEIGHT_DECAY = group_cfg.pop("weight_decay", BASE_WEIGHT_DECAY)

        kwargs = {
            "params": params,
            "lr": BASE_LEARNING_RATE * LR_SCALE,
            "weight_decay": WEIGHT_DECAY,
            **group_cfg,
        }
        args.append(kwargs)
        
        for p in params:
            assigned_params.add(p)

    default_params = [
        p for p in model.parameters()
        if p.requires_grad and p not in assigned_params
    ]

    if default_params:
        kwargs = {
            "params": default_params,
            "lr": BASE_LEARNING_RATE,
            "weight_decay": BASE_WEIGHT_DECAY,
        }
        args.append(kwargs)

    opt_cls = OPTIMIZER_REGISTRY[NAME]

    return opt_cls(args)


def optimizer_builder(
    model: nn.Module,
    cfg: dict,
):
    CONDITION = [
        callable(getattr(model, "param_groups", None)),
        "param_groups" in cfg.keys(),
    ]

    return (
        _grouped_optimizer(model, cfg) 
        if all(CONDITION)
        else _base_optimizer(model, cfg)
    )