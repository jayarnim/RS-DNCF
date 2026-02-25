import torch.nn as nn
from .evaluator import Evaluator
from .predictor import Predictor
from .calculator import Calculator


def evaluator_builder(
    model: nn.Module, 
    cfg,
):
    kwargs = dict(
        model=model,
        schema=cfg.schema,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        cutoff=cfg.cutoff,
        schema=cfg.schema,
    )
    calculator = Calculator(**kwargs)
    
    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Evaluator(**kwargs)