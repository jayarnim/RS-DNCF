from dataclasses import dataclass
from typing import Literal
from .schema import SchemaCfg


@dataclass
class CandidatesCfg:
    schema: SchemaCfg


@dataclass
class SplitCfg:
    schema: SchemaCfg
    ratio: dict[str, int]
    min_rating: int
    filter_by: Literal["user", "item"]
    seed: int


@dataclass
class DataloaderCfg:
    schema: SchemaCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    ratio: dict[str, int]
    batch_size: int
    shuffle: bool
    seed: int


@dataclass
class InteractionsCfg:
    schema: SchemaCfg
    num_users: int
    num_items: int


@dataclass
class SelectorCfg:
    name: str
    params: dict


@dataclass
class HistoriesCfg:
    selector: SelectorCfg


@dataclass
class PipelineCfg:
    candidates: CandidatesCfg
    split: SplitCfg
    dataloader: DataloaderCfg
    interactions: InteractionsCfg
    histories: HistoriesCfg