import pandas as pd


def main(
    df: pd.DataFrame, 
    schema,
    percentaile: float=0.9,
):
    user_counts = df[schema.col_user].value_counts()
    item_counts = df[schema.col_item].value_counts()

    N_USERS = df[schema.col_user].nunique()
    N_ITEMS = df[schema.col_item].nunique()
    TOTAL_INTERACTION = len(df)
    DENSITY = df.shape[0] / (N_USERS * N_ITEMS)
    MAX_USER_INTERACTION = user_counts.max()
    MAX_ITEM_INTERACTION = item_counts.max()
    TOP_PERCENTAILE_USER_INTERACTION = user_counts.quantile(percentaile)
    TOP_PERCENTAILE_ITEM_INTERACTION = item_counts.quantile(percentaile)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"interaction density: {DENSITY * 100:.4f} %",
        f"max interaction of user: {MAX_USER_INTERACTION}",
        f"max interaction of item: {MAX_ITEM_INTERACTION}",
        f"top {(1-percentaile) * 100:.1f} % interaction of user: {TOP_PERCENTAILE_USER_INTERACTION:.1f}",
        f"top {(1-percentaile) * 100:.1f} % interaction of item: {TOP_PERCENTAILE_ITEM_INTERACTION:.1f}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        sep="\n",
    )
