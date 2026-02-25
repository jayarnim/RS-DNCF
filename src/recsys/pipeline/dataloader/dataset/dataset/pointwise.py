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
        user, pos = self.pairs[idx // (1 + self.neg_ratio)]

        DECISION = (idx % (1 + self.neg_ratio) == 0)

        if DECISION:
            return user, pos, 1
        else:
            neg = self.sampler(user)
            return user, neg, 0

    def _set_up_components(self):
        obj = zip(
            self.df[self.schema.col_user], 
            self.df[self.schema.col_item],
        )
        self.pairs = list(obj)
        self.num_samples = len(self.pairs) * (1 + self.neg_ratio)