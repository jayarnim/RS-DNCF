import pandas as pd
import torch
import torch.nn as nn
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)
from .predictor import evaluation_predictor
from .metrics_computer import metrics_computer


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(
        self,
        model: nn.Module, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        self.model = model.to(DEVICE)
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

    def predict(
        self,
        tst_loader: torch.utils.data.dataloader.DataLoader,
    ):
        kwargs = dict(
            model=self.model,
            tst_loader=tst_loader,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        return evaluation_predictor(**kwargs)

    def metrics(
        self,
        result: pd.DataFrame,
        k_list: list,
    ):
        kwargs = dict(
            result=result,
            k_list=k_list,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        return metrics_computer(**kwargs)