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

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

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

        kwargs = dict(
            tensors=(pred_vector_dgmf, pred_vector_dmlp), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)
        logit = self.logit_layer(pred_vector).squeeze(-1)

        return logit

    def _init_layers(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors//2,
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

        kwargs = dict(
            in_features=self.n_factors//2 + self.hidden[-1],
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)