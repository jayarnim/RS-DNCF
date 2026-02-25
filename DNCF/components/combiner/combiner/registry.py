from .sum import ElementwiseSum
from .mean import ElementwiseMean
from .cat import Concatenation
from .att import Attention


COMBINER_REGISTRY = {
    "sum": ElementwiseSum,
    "mean": ElementwiseMean,
    "cat": Concatenation,
    "att": Attention,
}