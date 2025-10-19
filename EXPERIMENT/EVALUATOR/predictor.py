from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation_predictor(
    model: nn.Module, 
    tst_loader: torch.utils.data.dataloader.DataLoader,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    # evaluation
    model.eval()

    # to save result
    user_idx_list = []
    item_idx_list = []
    label_list = []
    pred_list = []

    iter_obj = tqdm(
        iterable=tst_loader, 
        desc=f"TST",
    )

    for user_idx, item_idx, label in iter_obj:
        # to gpu
        kwargs = dict(
            user_idx=user_idx.to(DEVICE),
            item_idx=item_idx.to(DEVICE),
        )
        
        # predict
        logit = model.predict(**kwargs)
        pred = torch.sigmoid(logit)

        # save
        user_idx_list.extend(user_idx.cpu().tolist())
        item_idx_list.extend(item_idx.cpu().tolist())
        label_list.extend(label.cpu().tolist())
        pred_list.extend(pred.cpu().tolist())

    # list -> df
    result = pd.DataFrame(
        {
            col_user: user_idx_list,
            col_item: item_idx_list,
            col_rating: label_list,
            col_prediction: pred_list,
        }
    )

    return result