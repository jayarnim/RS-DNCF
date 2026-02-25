from .schema import schema
from ..config.pipeline import (
    CandidatesCfg,
    SplitCfg,
    DataloaderCfg,
    InteractionsCfg,
    SelectorCfg,
    HistoriesCfg,
    PipelineCfg,
)


def candidates(cfg):
    return CandidatesCfg(
        schema=schema(cfg),
    )

def split(cfg):
    return SplitCfg(
        schema=schema(cfg), 
        ratio=cfg["split"]["ratio"], 
        min_rating=cfg["split"]["min_rating"], 
        filter_by=cfg["split"]["filter_by"], 
        seed=cfg["seed"],
    )

def dataloader(cfg):
    return DataloaderCfg(
        schema=schema(cfg),
        strategy=cfg["strategy"],
        ratio=cfg["dataloader"]["negatives"]["ratio"],
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
        seed=cfg["seed"],
    )

def interactions(cfg):
    return InteractionsCfg(
        schema=schema(cfg),
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
    )

def selector(cfg):
    return SelectorCfg(
        name=cfg["histories"]["selector"]["name"],
        params=cfg["histories"]["selector"].get("params") or dict(),
    )

def histories(cfg):
    return HistoriesCfg(
        selector=selector(cfg),
    )

def pipeline(cfg):
    return PipelineCfg(
        candidates=candidates(cfg),
        split=split(cfg),
        dataloader=dataloader(cfg),
        interactions=interactions(cfg),
        histories=histories(cfg),
    )