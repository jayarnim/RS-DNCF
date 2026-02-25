from .idx import IDXEmbedding
from .history import HistoryEmbedding


EMBEDDING_REGISTRY = {
    "idx": IDXEmbedding,
    "history": HistoryEmbedding,
}