import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self, 
        dim: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self, 
        id_emb: torch.Tensor, 
        hist_emb: torch.Tensor,
    ):
        id_emb_score = self.mlp(id_emb)
        hist_emb_score = self.mlp(hist_emb)
        weight = (
            torch.exp(id_emb_score) 
            / (
                torch.exp(id_emb_score) + torch.exp(hist_emb_score)
            )
        )
        return weight * id_emb + (1-weight) * hist_emb