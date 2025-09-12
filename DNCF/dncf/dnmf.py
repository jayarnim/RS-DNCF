import torch
import torch.nn as nn
from . import dgmf, dmlp


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.interactions = interactions.to(self.device)

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector_dgmf = self.dgmf.gmf(user_idx, item_idx)
        pred_vector_dmlp = self.dmlp.ncf(user_idx, item_idx)
        pred_vector = torch.cat(
            tensors=(pred_vector_dgmf, pred_vector_dmlp), 
            dim=-1,
        )
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def _init_layers(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            interactions=self.interactions,
        )
        self.dgmf = dgmf.Module(**kwargs)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.dmlp = dmlp.Module(**kwargs)

        self.logit_layer = nn.Linear(
            in_features=self.n_factors + self.hidden[-1],
            out_features=1,
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 4)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 4}"
        assert CONDITION, ERROR_MESSAGE