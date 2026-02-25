import torch
from components.base import BaseModel
from .dgmf import DeepGeneralMatrixFactorization
from .dmlp import DeepMultiLayerPerceptron
from .layers.fusion import ConcatenationLayer
from .layers.prediction import ProjectionLayer


class DeepNeuralMatrixFactorization(BaseModel):
    def __init__(
        self,
        dgmf: DeepGeneralMatrixFactorization,
        dmlp: DeepMultiLayerPerceptron, 
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
        super().__init__(locals())

        # ENSEMBLE MODULES ==========
        self.dgmf = dgmf
        self.dmlp = dmlp
        self.pred_dim = dgmf.pred_dim + dmlp.pred_dim

        # FUSION ==========
        self.fusion = ConcatenationLayer()

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # ENSEMBLE LEARNING ==========
        args = (
            self.dgmf(user_idx, item_idx),
            self.dmlp(user_idx, item_idx),
        )
        # ENSEMBLE AGGREGATION ==========
        X_pred = self.fusion(*args)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit