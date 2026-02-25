import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(
        self,
        pos: torch.Tensor, 
        neg: torch.Tensor,
    ):
        diff = pos - neg
        return -F.logsigmoid(diff).mean()