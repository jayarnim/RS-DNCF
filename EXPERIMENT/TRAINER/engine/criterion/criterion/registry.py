from .pointwise import bce
from .pairwise import bpr
from .listwise import climf


POINTWISE_CRITERION_REGISTRY = {
    "bce": bce,
}

PAIRWISE_CRITERION_REGISTRY = {
    "bpr": bpr,
}

LISTWISE_CRITERION_REGISTRY = {
    "climf": climf,
}

CRITERION_REGISTRY = {
    "pointwise": POINTWISE_CRITERION_REGISTRY,
    "pairwise": PAIRWISE_CRITERION_REGISTRY,
    "listwise": LISTWISE_CRITERION_REGISTRY,
}