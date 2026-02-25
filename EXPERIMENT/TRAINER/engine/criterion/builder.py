from .criterion.registry import CRITERION_REGISTRY


def criterion_builder(
    strategy: str,
    criterion: str,
):
    return CRITERION_REGISTRY[strategy][criterion]