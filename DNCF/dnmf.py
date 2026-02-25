import torch
import torch.nn as nn
from . import dgmf, dmlp
from .components.scorer import LinearProjectionLayer


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
        self.predictive_dim = dgmf.predictive_dim + dmlp.predictive_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        predictive_vec_dgmf = self.dgmf(user_idx, item_idx)
        predictive_vec_dmlp = self.dmlp(user_idx, item_idx)

        # fusion
        kwargs = dict(
            tensors=(predictive_vec_dgmf, predictive_vec_dmlp), 
            dim=-1,
        )
        predictive_vec_fusion = torch.cat(**kwargs)

        return predictive_vec_fusion

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            input_dim=self.predictive_dim,
        )
        self.scorer = LinearProjectionLayer(**kwargs)