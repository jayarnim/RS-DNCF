from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import DGMFCfg, DMLPCfg, DNMFCfg


@dataclass
class Config:
    model: Union[DGMFCfg, DMLPCfg, DNMFCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["dgmf", "dmlp", "dnmf"]
    dataset: str
    seed: int