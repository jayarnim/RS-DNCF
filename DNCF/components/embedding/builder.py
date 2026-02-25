from .embedding.registry import EMBEDDING_REGISTRY


def embedding_builder(name, **kwargs):
    cls = EMBEDDING_REGISTRY[name]
    return cls(**kwargs)