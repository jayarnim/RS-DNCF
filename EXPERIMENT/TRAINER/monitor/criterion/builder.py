from .criterion.registry import CRITERION_REGISTRY


def criterion_builder(
    criterion: str,
):
    return CRITERION_REGISTRY[criterion]