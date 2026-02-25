from .model import model
from .schema import schema
from .pipeline import pipeline
from .trainer import trainer
from .evaluator import evaluator
from ..config.config import Config


def parser(cfg):
    return Config(
        model=model(cfg),
        schema=schema(cfg),
        pipeline=pipeline(cfg),
        trainer=trainer(cfg),
        evaluator=evaluator(cfg),
        model_cls=cfg["model"]["name"],
        dataset=cfg["data"]["name"],
        strategy=cfg["strategy"],
        seed=cfg["seed"],
    )