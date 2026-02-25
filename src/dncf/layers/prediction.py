import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_features=dim,
            out_features=1,
        )

    def forward(
        self, 
        X: torch.Tensor, 
    ) -> torch.Tensor:
        return self.linear(X).squeeze(-1)