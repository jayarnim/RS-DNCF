import torch
import torch.nn as nn
from .components.embedding import EmbeddingLayer
from .components.projection import ProjectionLayer
from .components.combiner.builder import build_combiner


class Module(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        num_factors: int,
        combiner: str,
    ):
        """
        Dual-embedding based neural collaborative filtering for recommender systems (He et al., 2021)
        -----
        Implements the base structure of Dual General Matrix Factorization (DGMF),
        MF & dual embedding based latent factor model,
        sub-module of Dual Neural Matrix Factorization (DNMF)
        to learn low-rank linear represenation.

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
            combiner (str):
                function type that combines identifier embeddings and history embeddings.
                (e.g. `sum`, `mean`, `cat`, `att`)
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
        self.combiner = combiner
        self.matching_dim = (
            num_factors * 2
            if combiner=="cat"
            else num_factors
        )

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # Embedding
        user_emb_id, item_emb_id = self.embedding(user_idx, item_idx)
        user_emb_hist, item_emb_hist = self.projection(user_idx, item_idx)
        
        # Combination
        user_emb_comb = self.combiner_user(user_emb_id, user_emb_hist)
        item_emb_comb = self.combiner_item(item_emb_id, item_emb_hist)
        
        # Matching
        matching_vec = user_emb_comb * item_emb_comb
        
        return matching_vec

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
        self._create_components()
        self._create_layers()

    def _create_components(self):
        kwargs = dict(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.num_factors,
        )
        self.embedding = EmbeddingLayer(**kwargs)

        kwargs = dict(
            interactions=self.interactions,
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.num_factors,
        )
        self.projection = ProjectionLayer(**kwargs)

        kwargs = dict(
            name=self.combiner,
            dim=self.num_factors,
        )
        self.combiner_user = build_combiner(**kwargs)
        self.combiner_item = build_combiner(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.matching_dim,
            out_features=1, 
        )
        self.prediction = nn.Linear(**kwargs)