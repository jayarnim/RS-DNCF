from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main(
    df: pd.DataFrame,
    schema,
    col_user: str,
    col_item: str,
    col_rating: Optional[str]=None,
    col_timestamp: Optional[str]=None,
):
    ORIGIN_COL_LIST = [col_user, col_item]
    RENAME_COL_LIST = [schema.col_user, schema.col_item]

    if col_rating is not None:
        ORIGIN_COL_LIST.append(col_rating)
        RENAME_COL_LIST.append(schema.col_rating)

    if col_timestamp is not None:
        ORIGIN_COL_LIST.append(col_timestamp)
        RENAME_COL_LIST.append(schema.col_timestamp)

    RENAMES = dict(zip(ORIGIN_COL_LIST, RENAME_COL_LIST))
    
    df = df[ORIGIN_COL_LIST]
    df = df.rename(columns=RENAMES)

    encoder = dict(
        user=LabelEncoder(),
        item=LabelEncoder(),
    )
    df[schema.col_user] = encoder["user"].fit_transform(df[schema.col_user]) + 2
    df[schema.col_item] = encoder["item"].fit_transform(df[schema.col_item]) + 2

    return df