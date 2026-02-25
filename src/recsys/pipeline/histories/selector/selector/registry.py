from .default import DefaultSelector
from .freq import FreqSelector
from .tfidf import TFIDFSelector


SELECTOR_REGISTRY = {
    "default": DefaultSelector,
    "freq": FreqSelector,
    "tfidf": TFIDFSelector,
}