from .schema import schema
from ..config.evaluator import EvaluatorCfg


def evaluator(cfg):
    return EvaluatorCfg(
        schema=schema(cfg),
        cutoff=cfg["evaluator"]["cutoff"],
    )