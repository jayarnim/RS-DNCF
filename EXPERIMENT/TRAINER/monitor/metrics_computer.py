import pandas as pd
from ...constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)


class MetricsComputer:
    def __init__(
        self, 
        criterion,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
        k: int=DEFAULT_K,
    ):
        self.criterion = criterion
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.k = k

    def __call__(
        self,
        result: pd.DataFrame,
    ):
        rating_true, rating_pred = self._true_pred_seperator(result)
        score = self._top_k_evaluator(rating_true, rating_pred)
        return score

    def _true_pred_seperator(self, result):
        TRUE_COL_LIST = [self.col_user, self.col_item, self.col_rating]
        PRED_COL_LIST = [self.col_user, self.col_item, self.col_prediction]

        rating_true = (
            result[TRUE_COL_LIST]
            [result[self.col_rating]==1]
            .sort_values(
                by=self.col_user, 
                ascending=True,
            )
        )

        rating_pred = (
            result[PRED_COL_LIST]
            .sort_values(
                by=[self.col_user, self.col_prediction], 
                ascending=[True, False], 
                kind='stable',
            ).groupby(self.col_user)
        )

        return rating_true, rating_pred

    def _top_k_evaluator(self, rating_true, rating_pred):
        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred.head(self.k),
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
            k=self.k,
        )
        return self.criterion(**kwargs)