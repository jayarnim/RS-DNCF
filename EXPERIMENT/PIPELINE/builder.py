import pandas as pd
from .dataloader.builder import dataloader_builder
from .histories.builder import histories_builder
from .interactions.interactions import interactions_generator
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


def pipeline_builder(
    strategy: str,
    df: pd.DataFrame, 
    cfg: dict,
    seed: int,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    NUM_USERS = df[col_user].nunique()
    NUM_ITEMS = df[col_item].nunique()

    kwargs = dict(
        strategy=strategy,
        df=df,
        cfg=cfg["dataloader"],
        seed=seed,
        col_user=col_user,
        col_item=col_item,
    )
    dataloaders = dataloader_builder(**kwargs)

    df_trn = dataloaders["trn"].dataset.df

    kwargs = dict(
        df=df_trn,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        col_user=col_user,
        col_item=col_item,
    )
    interactions = interactions_generator(**kwargs)

    kwargs = dict(
        interactions=interactions,
        cfg=cfg["histories"],
    )
    histories = histories_builder(**kwargs)

    return dataloaders, interactions, histories