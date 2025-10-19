import pandas as pd
from ..constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)
from ..msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)


def _true_pred_seperator(result, col_user, col_item, col_rating, col_prediction):
    TRUE_COL_LIST = [col_user, col_item, col_rating]
    PRED_COL_LIST = [col_user, col_item, col_prediction]

    rating_true = (
        result[TRUE_COL_LIST]
        [result[col_rating]==1]
        .sort_values(
            by=col_user, 
            ascending=True,
        )
    )

    rating_pred = (
        result[PRED_COL_LIST]
        .sort_values(
            by=[col_user, col_prediction], 
            ascending=[True, False], 
            kind='stable',
        ).groupby(col_user)
    )

    return rating_true, rating_pred


def _top_k_evaluator(rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, k):
    kwargs = dict(
        rating_true=rating_true,
        rating_pred=rating_pred.head(k),
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )

    return dict(
        k=k,
        hit_ratio=hit_ratio_at_k(**kwargs), 
        precision=precision_at_k(**kwargs), 
        recall=recall_at_k(**kwargs), 
        map=map_at_k(**kwargs), 
        ndcg=ndcg_at_k(**kwargs),
    )


def metrics_computer(
    result: pd.DataFrame,
    k_list: list,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    kwargs = dict(
        result=result,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    rating_true, rating_pred = _true_pred_seperator(**kwargs)

    eval_list = []

    for k in k_list:
        kwargs = dict(
            rating_true=rating_true, 
            rating_pred=rating_pred, 
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction=col_prediction,
            k=k,
        )
        eval = _top_k_evaluator(**kwargs)
        eval_list.append(eval)

    return pd.DataFrame(eval_list)