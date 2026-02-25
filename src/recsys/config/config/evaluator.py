from dataclasses import dataclass
from .schema import SchemaCfg


@dataclass
class EvaluatorCfg:
    cutoff: list[int]
    schema: SchemaCfg