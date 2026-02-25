import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ....constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class PairwiseDataset(Dataset):
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
        return self.total_samples * self.num_negatives

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx // self.num_negatives]
        neg = random.choice(self.candidates[user])
        return user, pos, neg

    def _set_up_components(self):
        zip_obj = zip(self.df[self.col_user], self.df[self.col_item])
        self.user_item_pairs = list(zip_obj)
        self.total_samples = len(self.user_item_pairs) * self.num_negatives


def _pairwise_collate_fn(batch):
    user_list, pos_list, neg_list = zip(*batch)

    user_batch = torch.tensor(user_list, dtype=torch.long)
    pos_batch = torch.tensor(pos_list, dtype=torch.long)
    neg_batch = torch.tensor(neg_list, dtype=torch.long)

    return user_batch, pos_batch, neg_batch


def pairwise_dataloader(
    df: pd.DataFrame,
    candidates: dict,
    num_negatives: int,
    batch_size: int,
    shuffle: bool=True,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
):
    kwargs = dict(
        origin=df,
        candidates=candidates, 
        num_negatives=num_negatives,
        col_user=col_user, 
        col_item=col_item,     
    )
    dataset = PairwiseDataset(**kwargs)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_pairwise_collate_fn,
    )
    dataloader = DataLoader(**kwargs)

    return dataloader