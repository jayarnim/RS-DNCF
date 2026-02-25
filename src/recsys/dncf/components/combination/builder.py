from .combination.registry import COMBINATION_REGISTRY


def combination_builder(name, **kwargs):
    cls = COMBINATION_REGISTRY[name]
    return cls(**kwargs)