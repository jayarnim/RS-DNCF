import torch
import torch.nn as nn


class Concatenation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        id_emb: torch.Tensor, 
        hist_emb: torch.Tensor,
    ):
        kwargs = dict(
            tensors=(id_emb, hist_emb), 
            dim=-1,
        )
        return torch.cat(**kwargs)