import pandas as pd
from torch.utils.data import Dataset


class CustomizedDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        sampler,
        schema,
    ):
        super().__init__()
        self.df = df
        self.sampler = sampler
        self.neg_ratio = sampler.ratio
        self.schema = schema
        self._set_up_components()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        user, pos = self.pairs[idx // self.neg_ratio]
        neg = self.sampler(user)
        return user, pos, neg

    def _set_up_components(self):
        zip_obj = zip(
            self.df[self.schema.col_user], 
            self.df[self.schema.col_item],
        )
        self.pairs = list(zip_obj)
        self.num_samples = len(self.pairs) * self.neg_ratio