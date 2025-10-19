import torch
import torch.nn as nn
from .trainer import Trainer
from .optimizer.builder import optimizer_builder
from .scheduler.builder import scheduler_builder
from .engine.builder import engine_builder
from .monitor.builder import monitor_builder
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer_builder(
    model: nn.Module,
    cfg: dict,
    strategy: str,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: str=DEFAULT_K,
):
    NUM_EPOCHS = cfg["trainer"]["num_epochs"]

    model = model.to(DEVICE)

    kwargs = dict(
        model=model,
        cfg=cfg["optimizer"],
    )
    optimizer = optimizer_builder(**kwargs)

    kwargs = dict(
        optimizer=optimizer,
        cfg=cfg["scheduler"],
    )
    scheduler = scheduler_builder(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=strategy,
        cfg=cfg["engine"],
    )
    engine = engine_builder(**kwargs)

    kwargs = dict(
        model=model,
        cfg=cfg["monitor"],
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )
    monitor = monitor_builder(**kwargs)

    kwargs = dict(
        model=model,
        engine=engine,
        monitor=monitor,
        num_epochs=NUM_EPOCHS,
    )
    trainer = Trainer(**kwargs)

    return trainer