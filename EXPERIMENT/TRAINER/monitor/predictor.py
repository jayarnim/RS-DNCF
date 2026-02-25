from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(
        self, 
        model: nn.Module, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        self.model = model
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

    @torch.no_grad()
    def __call__(
        self,
        dev_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
    ):
        # evaluation
        self.model.eval()

        # to save result
        user_idx_list = []
        item_idx_list = []
        label_list = []
        pred_list = []

        iter_obj = tqdm(
            iterable=dev_loader, 
            desc=f"EPOCH {epoch+1} DEV"
        )

        for user_idx, item_idx, label in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                item_idx=item_idx.to(DEVICE),
            )

            # predict
            logit = self.model.predict(**kwargs)
            pred = torch.sigmoid(logit)

            # to cpu & save
            user_idx_list.extend(user_idx.cpu().tolist())
            item_idx_list.extend(item_idx.cpu().tolist())
            label_list.extend(label.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        # list -> df
        result = pd.DataFrame(
            {
                self.col_user: user_idx_list,
                self.col_item: item_idx_list,
                self.col_rating: label_list,
                self.col_prediction: pred_list,
            }
        )

        return result