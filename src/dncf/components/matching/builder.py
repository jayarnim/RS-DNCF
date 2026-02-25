from .matching.registry import MATCHING_FN_REGISTRY


def matching_fn_builder(name, **kwargs):
    cls = MATCHING_FN_REGISTRY[name]
    return cls(**kwargs)