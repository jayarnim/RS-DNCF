import torch
from torch.nn.utils.rnn import pad_sequence
from .selector.registry import SELECTOR_REGISTRY


def histories_generator(
    interactions: torch.Tensor, 
    selector: str,
    max_hist: int,
):
    # drop padding idx
    interactions_unpadded = interactions[:-1, :-1]

    # padding idx
    num_anchor, num_target = interactions_unpadded.shape

    # select hist per anchor
    kwargs = dict(
        interactions=interactions_unpadded,
        max_hist=max_hist,
    )
    hist_indices = SELECTOR_REGISTRY[selector](**kwargs)

    # padding
    kwargs = dict(
        sequences=hist_indices, 
        batch_first=True, 
        padding_value=num_target,
    )
    hist_indices_padded = pad_sequence(**kwargs)

    return hist_indices_padded