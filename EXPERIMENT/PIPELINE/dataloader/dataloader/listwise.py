import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ....constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class ListwiseDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        candidates: dict,
        num_negatives: int,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.df = df
        self.candidates = candidates
        self.num_negatives = num_negatives
        self.col_user = col_user
        self.col_item = col_item

        self._set_up_components()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]
        kwargs = dict(
            population=self.candidates[user],
            k=self.num_negatives,
        )
        neg_list = random.sample(**kwargs)
        return user, pos, neg_list

    def _set_up_components(self):
        zip_obj = zip(self.df[self.col_user], self.df[self.col_item])
        self.user_item_pairs = list(zip_obj)
        self.total_samples = len(self.user_item_pairs)


def _listwise_collate_fn(batch):
    user_list, pos_list, neg_list = zip(*batch)
    
    user_batch = torch.tensor(user_list, dtype=torch.long)          # (B,)
    pos_batch  = torch.tensor(pos_list, dtype=torch.long)           # (B,)
    neg_batch  = torch.tensor(neg_list, dtype=torch.long)           # (B, N)
    
    return user_batch, pos_batch, neg_batch


def listwise_dataloader(
    df: pd.DataFrame,
    candidates: dict,
    num_negatives: int,
    batch_size: int,
    shuffle: bool=True,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
):
    kwargs = dict(
        df=df, 
        candidates=candidates,
        num_negatives=num_negatives,
        col_user=col_user, 
        col_item=col_item,     
    )
    dataset = ListwiseDataset(**kwargs)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_listwise_collate_fn,
    )
    dataloader = DataLoader(**kwargs)

    return dataloader