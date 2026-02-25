from ..config.schema import SchemaCfg
from ...msr.const import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)


def schema(*args):
    return SchemaCfg(
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_label=DEFAULT_LABEL_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    )