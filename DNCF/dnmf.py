import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        dgmf: nn.Module,
        dmlp: nn.Module, 
    ):
        """
        Dual-embedding based neural collaborative filtering for recommender systems (He et al., 2021)
        -----
        Implements the base structure of Dual Neural Matrix Factorization (DNMF),
        MF, MLP & dual embedding based latent factor model,
        combining a Dual General Matrix Factorization (DGMF) and a Dual Multi-Layer Perceptron (DMLP)
        to learn low-rank linear represenation & high-rank nonlinear user-item interactions.

        Args:
            dgmf (nn.Module)
            dmlp (nn.Module)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.dgmf = dgmf
        self.dmlp = dmlp
        self.matching_dim = dgmf.matching_dim + dmlp.matching_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        matching_vec_dgmf = self.dgmf(user_idx, item_idx)
        matching_vec_dmlp = self.dmlp(user_idx, item_idx)

        # fusion
        kwargs = dict(
            tensors=(matching_vec_dgmf, matching_vec_dmlp), 
            dim=-1,
        )
        matching_vec_fusion = torch.cat(**kwargs)

        return matching_vec_fusion

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.matching_dim,
            out_features=1,
        )
        self.prediction = nn.Linear(**kwargs)