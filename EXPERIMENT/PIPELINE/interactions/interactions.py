import torch
import pandas as pd
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


def interactions_generator(
    df: pd.DataFrame, 
    num_users: int,
    num_items: int,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
):
    kwargs = dict(
        size=(num_users + 1, num_items + 1),
        dtype=torch.int32,
    )
    user_item_matrix = torch.zeros(**kwargs)

    kwargs = dict(
        data=df[col_user].values, 
        dtype=torch.long,
    )
    user_indices = torch.tensor(**kwargs)
    
    kwargs = dict(
        data=df[col_item].values, 
        dtype=torch.long,
    )
    item_indices = torch.tensor(**kwargs)

    user_item_matrix[user_indices, item_indices] = 1

    return user_item_matrix