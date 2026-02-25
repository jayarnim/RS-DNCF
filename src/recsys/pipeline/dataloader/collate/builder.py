from .collate.registry import COLLATE_FN_REGISTRY


def collate_fn_builder(strategy):
    return COLLATE_FN_REGISTRY[strategy]