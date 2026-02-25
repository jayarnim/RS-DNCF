from dataclasses import dataclass


@dataclass
class DGMFCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class DMLPCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class DNMFCfg:
    dgmf: DGMFCfg
    dmlp: DMLPCfg