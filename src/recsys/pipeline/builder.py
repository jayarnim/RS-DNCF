import pandas as pd
from .candidates import negative_sample_candidates_builder
from .split import stratified_split_builder
from .interactions import interactions_builder
from .histories.builder import histories_builder
from .dataloader.builder import dataloader_builder


def pipeline_builder(
    df: pd.DataFrame, 
    cfg,
):
    kwargs = dict(
        df=df,
        cfg=cfg.candidates,
    )
    candidates = negative_sample_candidates_builder(**kwargs)

    kwargs = dict(
        df=df,
        cfg=cfg.split,
    )
    split = stratified_split_builder(**kwargs)

    kwargs = dict(
        split=split,
        candidates=candidates,
        cfg=cfg.dataloader,
    )
    dataloader = dataloader_builder(**kwargs)

    kwargs = dict(
        df=split["trn"],
        cfg=cfg.interactions,
    )
    interactions = interactions_builder(**kwargs)

    kwargs = dict(
        interactions=interactions,
        cfg=cfg.histories,
    )
    histories = histories_builder(**kwargs)

    return dataloader, interactions, histories