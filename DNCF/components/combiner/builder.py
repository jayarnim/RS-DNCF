from .combiner.registry import COMBINER_REGISTRY


def combiner_builder(name, **kwargs):
    cls = COMBINER_REGISTRY[name]
    return cls(**kwargs)