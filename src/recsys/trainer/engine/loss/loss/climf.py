import torch
import torch.nn as nn
import torch.nn.functional as F


class CLiMF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(
        self,
        pos: torch.Tensor, 
        neg: torch.Tensor,
    ):
        diff = neg - pos.unsqueeze(1)
        max_pos_term  = F.logsigmoid(pos)
        min_diff_term = F.logsigmoid(-diff).sum(dim=1)
        return -(max_pos_term + min_diff_term).mean()