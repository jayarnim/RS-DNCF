import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.combination.builder import combination_builder
from .components.matching.builder import matching_fn_builder
from .components.prediction import ProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        num_factors: int,
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
            interactions (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
            num_users (int): 
                total number of users in the dataset, U.
            num_items (int): 
                total number of items in the dataset, I.
            num_factors (int): 
                dimensionality of user and item latent factor vectors, K.
            hidden_dim (list): 
                layer dimensions for the MLP-based matching function. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pred_dim = hidden_dim[-1]

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # Embedding
        user_emb_id, item_emb_id = self.embedding["idx"](user_idx, item_idx)
        user_emb_hist, item_emb_hist = self.embedding["history"](user_idx, item_idx)

        # Combination
        user_emb_comb = self.combination["user"](user_emb_id, user_emb_hist)
        item_emb_comb = self.combination["item"](item_emb_id, item_emb_hist)

        # Matching
        X_pred = self.matching(user_emb_comb, item_emb_comb)

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
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            name="idx",
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.num_factors,
        )
        idx = embedding_builder(**kwargs)

        kwargs = dict(
            name="history",
            interactions=self.interactions,
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.num_factors,
        )
        history = embedding_builder(**kwargs)

        components = dict(
            idx=idx,
            history=history,
        )
        self.embedding = nn.ModuleDict(components)

        kwargs = dict(
            name="cat",
            dim=self.num_factors,
        )
        components = dict(
            user=combination_builder(**kwargs),
            item=combination_builder(**kwargs),
        )
        self.combination = nn.ModuleDict(components)

        kwargs = dict(
            name="ncf",
            input_dim=self.num_factors*4,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            dim=self.hidden_dim[-1],
        )
        self.prediction = ProjectionLayer(**kwargs)