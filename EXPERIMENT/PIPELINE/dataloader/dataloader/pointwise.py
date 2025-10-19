import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ....constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class PointwiseDataset(Dataset):
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
        decision = (idx % (1 + self.num_negatives) == 0)

        if decision==True:
            user, pos = self.user_item_pairs[idx // (1 + self.num_negatives)]
            return user, pos, 1
        else:
            user, _ = self.user_item_pairs[idx // (1 + self.num_negatives)]
            neg = random.choice(self.candidates[user])
            return user, neg, 0

    def _set_up_components(self):
        zip_obj = zip(self.df[self.col_user], self.df[self.col_item])
        self.user_item_pairs = list(zip_obj)
        self.total_samples = len(self.user_item_pairs) * (1 + self.num_negatives)


def _pointwise_collate_fn(batch):
    user_list, item_list, label_list = zip(*batch)
    
    user_batch = torch.tensor(user_list, dtype=torch.long)
    item_batch = torch.tensor(item_list, dtype=torch.long)
    label_batch = torch.tensor(label_list, dtype=torch.float32)
    
    return user_batch, item_batch, label_batch


def pointwise_dataloader(
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
    dataset = PointwiseDataset(**kwargs)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_pointwise_collate_fn,
    )
    dataloader = DataLoader(**kwargs)

    return dataloader