import torch
from .histories import histories_generator


def histories_builder(
    interactions: torch.Tensor, 
    cfg: dict,
):
    histories = dict()

    USER_SELECTOR = cfg["user"]["selector"]
    USER_MAX_HIST = cfg["user"]["max_hist"]

    kwargs = dict(
        interactions=interactions,
        selector=USER_SELECTOR,
        max_hist=USER_MAX_HIST,
    )
    histories["user"] = histories_generator(**kwargs)

    ITEM_SELECTOR = cfg["item"]["selector"]
    ITEM_MAX_HIST = cfg["item"]["max_hist"]

    kwargs = dict(
        interactions=interactions.T,
        selector=ITEM_SELECTOR,
        max_hist=ITEM_MAX_HIST,
    )
    histories["item"] = histories_generator(**kwargs)

    return histories