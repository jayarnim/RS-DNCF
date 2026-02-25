import torch
import torch.nn as nn
from components.interactions import Interactions
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.combination import build as build_comb_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class DeepMultiLayerPerceptron(BaseModel):
    def __init__(
        self,
        interactions: Interactions, 
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Dual-embedding based neural collaborative filtering for recommender systems (He et al., 2021)
        -----
        Implements the base structure of Dual Multi-Layer Perceptron (DMLP),
        MLP & dual embedding based latent factor model,
        sub-module of Dual Neural Matrix Factorization (DNMF)
        to learn high-rank nonlinear user-item interactions.

        Args:
            interactions (Interactions): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+2, I+2])
            num_users (int): 
                total number of users in the dataset, U.
            num_items (int): 
                total number of items in the dataset, I.
            embedding_dim (int): 
                dimensionality of user and item latent factor vectors, K.
            hidden_dim (list): 
                layer dimensions for the MLP-based matching function. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__(locals())

        self.pred_dim = hidden_dim[-1]

        # USER-ITEM INTERACTION MAT. VIEWER ==========
        self.interactions = interactions

        # IDX & HIST EMBEDDING ==========
        kwargs = dict(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )
        components = dict(
            idx=build_embedding_layer(
                name="idx",
                **kwargs,
            ),
            history=build_embedding_layer(
                name="history",
                **kwargs,
            ),
        )
        self.embedding = nn.ModuleDict(components)

        # EMBEDDING COMBINATION ==========
        kwargs = dict(
            name="cat",
            dim=embedding_dim,
        )
        components = dict(
            user=build_comb_layer(**kwargs),
            item=build_comb_layer(**kwargs),
        )
        self.combination = nn.ModuleDict(components)

        # MATCHING FUNCTION LEARNING ==========
        self.matching = build_matching_layer(
            name="ncf",
            embedding_dim=embedding_dim*2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH USER-ITEM INTERACION MAT. ==========
        user_vec, item_vec = self.interactions(user_idx, item_idx)
        # EMBEDDINGS ==========
        user_emb_idx, item_emb_idx = self.embedding["idx"](user_idx, item_idx)
        user_emb_hist, item_emb_hist = self.embedding["history"](user_vec, item_vec)
        # COMBINATION ==========
        user_emb_comb = self.combination["user"](user_emb_idx, user_emb_hist)
        item_emb_comb = self.combination["item"](item_emb_idx, item_emb_hist)
        # MATCHING FUNCTION LEARNING ==========
        X_pred = self.matching(user_emb_comb, item_emb_comb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
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
