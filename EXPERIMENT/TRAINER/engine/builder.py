import torch.nn as nn
import torch.optim as optim
from .engine.registry import ENGINE_REGISTRY
from .criterion.builder import criterion_builder


def engine_builder(
    strategy: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: dict,
):
    CRITERION = cfg["criterion"]

    kwargs = dict(
        strategy=strategy,
        criterion=CRITERION,
    )
    criterion = criterion_builder(**kwargs)

    cls = ENGINE_REGISTRY[strategy]

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
    return cls(**kwargs)
