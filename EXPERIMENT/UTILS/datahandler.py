from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


def rename_columns(
    df: pd.DataFrame,
    col_user: str,
    col_item: str,
    col_rating: Optional[str]=None,
    col_timestamp: Optional[str]=None,
):
    ORIGIN_COL_LIST = [col_user, col_item]
    RENAME_COL_LIST = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]

    if col_rating is not None:
        ORIGIN_COL_LIST.append(col_rating)
        RENAME_COL_LIST.append(DEFAULT_RATING_COL)

    if col_timestamp is not None:
        ORIGIN_COL_LIST.append(col_timestamp)
        RENAME_COL_LIST.append(DEFAULT_TIMESTAMP_COL)

    RENAMES = dict(zip(ORIGIN_COL_LIST, RENAME_COL_LIST))
    
    df = df[ORIGIN_COL_LIST]
    df = df.rename(columns=RENAMES)

    return df


def label_encoding(
    df: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    user_encoder = LabelEncoder()
    df[col_user] = user_encoder.fit_transform(df[col_user])
    user_label = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
    
    item_encoder = LabelEncoder()
    df[col_item] = item_encoder.fit_transform(df[col_item])
    item_label = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))

    return df, user_label, item_label


def description(
    df: pd.DataFrame, 
    percentaile: float=0.9,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    user_counts = df[col_user].value_counts()
    item_counts = df[col_item].value_counts()

    N_USERS = df[col_user].nunique()
    N_ITEMS = df[col_item].nunique()
    TOTAL_INTERACTION = len(df)
    DENSITY = df.shape[0] / (N_USERS * N_ITEMS)
    MAX_USER_INTERACTION = user_counts.max()
    MAX_ITEM_INTERACTION = item_counts.max()
    TOP_PERCENTAILE_USER_INTERACTION = user_counts.quantile(percentaile)
    TOP_PERCENTAILE_ITEM_INTERACTION = item_counts.quantile(percentaile)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"interaction density: {DENSITY * 100:.4f} %",
        f"max interaction of user: {MAX_USER_INTERACTION}",
        f"max interaction of item: {MAX_ITEM_INTERACTION}",
        f"top {(1-percentaile) * 100:.1f} % interaction of user: {TOP_PERCENTAILE_USER_INTERACTION:.1f}",
        f"top {(1-percentaile) * 100:.1f} % interaction of item: {TOP_PERCENTAILE_ITEM_INTERACTION:.1f}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        sep="\n",
    )
