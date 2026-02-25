from . import (
    pointwise,
    pairwise,
    listwise,
)


DATASET_REGISTRY = {
    "pointwise": pointwise.CustomizedDataset,
    "pairwise": pairwise.CustomizedDataset,
    "listwise": listwise.CustomizedDataset,
    "msr": pointwise.CustomizedDataset,
}