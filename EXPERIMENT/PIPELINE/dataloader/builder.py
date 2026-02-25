import pandas as pd
from ...msr.python_splitters import python_stratified_split
from .dataloader.registry import DATALOADER_REGISTRY
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


def _data_stratified_splitter(df, split_ratio, seed, col_user, col_item):
    split_type = list(split_ratio.keys())
    split_size = list(split_ratio.values())

    # for dev data set
    dev = (
        df
        .groupby(col_user)
        .sample(n=1, random_state=seed)
        .sort_values(by=col_user)
        .reset_index(drop=True)
    )

    # for trn, val, tst data set
    trn_val_tst = (
        df[~df[[col_user, col_item]]
        .apply(tuple, axis=1)
        .isin(set(dev[[col_user, col_item]]
        .apply(tuple, axis=1)))]
        .reset_index(drop=True)
    )

    # trn_val_tst -> [trn, val, tst]
    kwargs = dict(
        data=trn_val_tst,
        ratio=split_size,
        col_user=col_user,
        col_item=col_item,
        seed=seed,
    )
    split_list = python_stratified_split(**kwargs)

    split_dict = dict(zip(split_type, split_list))    
    split_dict["dev"] = dev

    return split_dict

def _candidates_generator(df, col_user, col_item):
    user_list = sorted(df[col_user].unique())
    item_list = sorted(df[col_item].unique())

    pos_per_user = {
        user: set(df.loc[df[col_user]==user, col_item].tolist())
        for user in user_list
    }

    neg_per_user = {
        user: list(set(item_list) - pos_per_user[user])
        for user in user_list
    }

    return neg_per_user

def _dataloader_generator(strategy, split_dict, candidates, num_negatives, batch_size, shuffle, col_user, col_item):
    dataloader_dict = dict()

    for k, v in split_dict.items():
        kwargs = dict(
            df=v, 
            candidates=candidates,
            num_negatives=(
                num_negatives["opt"] 
                if k in ["trn", "val"]
                else num_negatives["eval"]
            ), 
            batch_size=batch_size, 
            shuffle=shuffle,
            col_user=col_user,
            col_item=col_item,
        )
        dataloader = (
            DATALOADER_REGISTRY[strategy](**kwargs)
            if k in ["trn", "val"] 
            else DATALOADER_REGISTRY["pointwise"](**kwargs)
        )
        dataloader_dict[k] = dataloader

    return dataloader_dict


def dataloader_builder(
    strategy: str,
    df: pd.DataFrame,
    cfg: dict,
    seed: int,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    SPLIT_RATIO = cfg["split_ratio"]
    NUM_NEGATIVES = cfg["num_negatives"]
    BATCH_SIZE = cfg["batch_size"]
    SHUFFLE = cfg["shuffle"]
    
    # split original data
    kwargs = dict(
        df=df,
        split_ratio=SPLIT_RATIO,
        seed=seed,
        col_user=col_user,
        col_item=col_item,
    )
    split_dict = _data_stratified_splitter(**kwargs)

    kwargs = dict(
        df=df,
        col_user=col_user,
        col_item=col_item,
    )
    candidates = _candidates_generator(**kwargs)

    # generate data loaders
    kwargs = dict(
        strategy=strategy,
        split_dict=split_dict, 
        candidates=candidates,
        num_negatives=NUM_NEGATIVES, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE,
        col_user=col_user,
        col_item=col_item,
    )
    dataloader_dict = _dataloader_generator(**kwargs)

    return dataloader_dict