import torch
import torch.nn as nn
from ..aggregation import Concatenation


class NeuralCollaborativeFilteringLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        X = self.aggregation(user_emb, item_emb)
        return self.mlp(X)

    def _set_up_components(self):
        self._create_components()
        self._create_layers()

    def _create_components(self):
        self.aggregation = Concatenation()

    def _create_layers(self):
        components = list(self._yield_linear_block(self.hidden_dim))
        self.mlp = nn.Sequential(*components)

    def _yield_linear_block(self, hidden_dim):
        IN_FEATRUES = self.input_dim
        
        for OUT_FEATURES in hidden_dim:
            yield nn.Sequential(
                nn.Linear(IN_FEATRUES, OUT_FEATURES),
                nn.LayerNorm(OUT_FEATURES),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            IN_FEATRUES = OUT_FEATURES