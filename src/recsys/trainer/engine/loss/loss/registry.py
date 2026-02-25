from .bce import BCE
from .bpr import BPR
from .climf import CLiMF


LOSS_FN_REGISTRY = {
    "bce": BCE,
    "bpr": BPR,
    "climf": CLiMF,
}