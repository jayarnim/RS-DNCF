from dataclasses import dataclass


@dataclass
class DGMFCfg:
    num_users: int
    num_items: int
    num_factors: int
    combiner: str


@dataclass
class DMLPCfg:
    num_users: int
    num_items: int
    num_factors: int
    hidden_dim: list
    dropout: float


@dataclass
class DNMFCfg:
    dgmf: DGMFCfg
    dmlp: DMLPCfg