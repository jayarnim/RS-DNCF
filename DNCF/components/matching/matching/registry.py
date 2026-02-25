from .mf import MatrixFactorizationLayer
from .ncf import NeuralCollaborativeFilteringLayer


MATCHING_FN_REGISTRY = {
    "mf": MatrixFactorizationLayer,
    "ncf": NeuralCollaborativeFilteringLayer,
}