import torch
import torch.nn as nn


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
        kwargs = dict(
            tensors=(user_emb, item_emb), 
            dim=-1,
        )
        concat = torch.cat(**kwargs)
        predictive_vec = self.mlp(concat)
        return predictive_vec

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )
        components = list(self._yield_linear_block(**kwargs))
        self.mlp = nn.Sequential(*components)

    def _yield_linear_block(self, input_dim, hidden_dim):
        idx = 0
        while idx < len(hidden_dim):
            INPUT_DIM = (
                input_dim
                if idx==0
                else hidden_dim[idx-1]
            )
            yield nn.Sequential(
                nn.Linear(INPUT_DIM, hidden_dim[idx]),
                nn.LayerNorm(hidden_dim[idx]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            idx += 1