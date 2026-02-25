from . import (
    pointwise,
    pairwise,
    listwise,
)


COLLATE_FN_REGISTRY = {
    "pointwise": pointwise.collate_fn,
    "pairwise": pairwise.collate_fn,
    "listwise": listwise.collate_fn,
    "msr": pointwise.collate_fn,
}