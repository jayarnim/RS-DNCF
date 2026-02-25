import pandas as pd
from typing import Literal
from ..msr.python_splitters import python_stratified_split


def stratified_split_builder(
    df: pd.DataFrame, 
    cfg, 
):
    split_type  = list(cfg.ratio.keys())
    split_ratio = list(cfg.ratio.values())

    # trn_val_tst -> [trn, val, tst]
    kwargs = dict(
        data=df,
        ratio=split_ratio,
        min_rating=cfg.min_rating,
        filter_by=cfg.filter_by,
        seed=cfg.seed,
        col_user=cfg.schema.col_user,
        col_item=cfg.schema.col_item,
    )
    split_list = python_stratified_split(**kwargs)

    split_dict = dict(zip(split_type, split_list))

    return split_dict