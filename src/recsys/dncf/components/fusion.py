import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args):
        kwargs = dict(
            tensors=args, 
            dim=-1,
        )
        return torch.cat(**kwargs)