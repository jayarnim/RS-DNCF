from .combiner.registry import COMBINER_REGISTRY


def build_combiner(name, dim):
    return (
        COMBINER_REGISTRY[name](dim)
        if name=="att"
        else COMBINER_REGISTRY[name]()
    )