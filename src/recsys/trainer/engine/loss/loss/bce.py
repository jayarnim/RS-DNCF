import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(
        self,
        logit: torch.Tensor, 
        label: torch.Tensor,
    ):
        return F.binary_cross_entropy_with_logits(input=logit, target=label)