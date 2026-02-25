import torch
import torch.nn as nn


class LinearProjectionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        predictive_vec: torch.Tensor, 
    ):
        logit = self.linear(predictive_vec).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.input_dim,
            out_features=1,
        )
        self.linear = nn.Linear(**kwargs)