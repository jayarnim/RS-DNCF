import torch.nn as nn
from .monitor import Monitor
from .early_stopper import EarlyStopper
from .metrics_computer import MetricsComputer
from .predictor import Predictor
from .criterion.builder import criterion_builder
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)


def monitor_builder(
    model: nn.Module,
    cfg: dict,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: str=DEFAULT_K,
):
    CRITERION = cfg["criterion"]
    DELTA = cfg["delta"]
    PATIENCE = cfg["patience"]
    WARMUP = cfg["warmup"]

    kwargs = dict(
        model=model,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        criterion=CRITERION,
    )
    criterion = criterion_builder(**kwargs)

    kwargs = dict(
        criterion=criterion,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )
    metrics_computer = MetricsComputer(**kwargs)

    kwargs = dict(
        delta=DELTA,
        patience=PATIENCE,
        warmup=WARMUP,
    )
    early_stopper = EarlyStopper(**kwargs)

    kwargs = dict(
        model=model,
        predictor=predictor,
        metrics_computer=metrics_computer,
        early_stopper=early_stopper,
    )
    monitor = Monitor(**kwargs)

    return monitor