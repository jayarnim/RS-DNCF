import torch.nn as nn
from .monitor import Monitor
from .predictor import Predictor
from .calculator import Calculator
from .early_stop import EarlyStopping
from .metric import METRIC_REGISTRY


def monitor_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        model=model,
        schema=cfg.schema,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        criterion=METRIC_REGISTRY[cfg.metric],
        k=cfg.k,
        schema=cfg.schema,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        delta=cfg.delta,
        patience=cfg.patience,
        warmup=cfg.warmup,
    )
    early_stop = EarlyStopping(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
        early_stop=early_stop,
    )
    return Monitor(**kwargs)