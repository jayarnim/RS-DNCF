from dataclasses import dataclass
from ...msr.const import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)


@dataclass
class SchemaCfg:
    col_user: str=DEFAULT_USER_COL
    col_item: str=DEFAULT_ITEM_COL
    col_rating: str=DEFAULT_RATING_COL
    col_label: str=DEFAULT_LABEL_COL
    col_timestamp: str=DEFAULT_TIMESTAMP_COL
    col_prediction: str=DEFAULT_PREDICTION_COL