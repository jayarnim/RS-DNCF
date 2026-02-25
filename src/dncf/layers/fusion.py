import torch
import torch.nn as nn


class ConcatenationLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        return torch.cat(
            tensors=args, 
            dim=-1,
        )