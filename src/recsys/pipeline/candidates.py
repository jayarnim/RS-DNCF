import pandas as pd


def negative_sample_candidates_builder(
    df: pd.DataFrame, 
    cfg,
):
    user_list = sorted(df[cfg.schema.col_user].unique())
    item_list = sorted(df[cfg.schema.col_item].unique())

    pos_per_user = {
        user: set(df.loc[df[cfg.schema.col_user]==user, cfg.schema.col_item].tolist())
        for user in user_list
    }

    neg_per_user = {
        user: list(set(item_list) - pos_per_user[user])
        for user in user_list
    }

    return neg_per_user