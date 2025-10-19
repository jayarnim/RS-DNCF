import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .engine.registry import ENGINE_REGISTRY
from .criterion.builder import criterion_builder


def engine_builder(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    cfg: dict,
    strategy: str,
):
    CRITERION = cfg["criterion"]

    kwargs = dict(
        strategy=strategy,
        criterion=CRITERION,
    )
    criterion = criterion_builder(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )
    return ENGINE_REGISTRY[strategy](**kwargs)
