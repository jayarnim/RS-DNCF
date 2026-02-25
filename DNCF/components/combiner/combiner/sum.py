import torch
import torch.nn as nn


class ElementwiseSum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        id_emb: torch.Tensor, 
        hist_emb: torch.Tensor,
    ):
        return id_emb + hist_emb