import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.combiner.builder import combiner_builder
from .components.matching.builder import matching_fn_builder
from .components.scorer import LinearProjectionLayer


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
        self.predictive_dim = (
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
        user_emb_id, item_emb_id = self.idx(user_idx, item_idx)
        user_emb_hist, item_emb_hist = self.history(user_idx, item_idx)
        
        # Combination
        user_emb_comb = self.combiner_user(user_emb_id, user_emb_hist)
        item_emb_comb = self.combiner_item(item_emb_id, item_emb_hist)
        
        # Matching
        predictive_vec = self.matching(user_emb_comb, item_emb_comb)
        
        return predictive_vec

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
            name="idx",
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.num_factors,
        )
        self.idx = embedding_builder(**kwargs)

        kwargs = dict(
            name="history",
            interactions=self.interactions,
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.num_factors,
        )
        self.history = embedding_builder(**kwargs)

        kwargs = dict(
            name=self.combiner,
            dim=self.num_factors,
        )
        self.combiner_user = combiner_builder(**kwargs)
        self.combiner_item = combiner_builder(**kwargs)

        kwargs = dict(
            name="mf",
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            input_dim=(
                self.num_factors*2
                if self.combiner=="cat"
                else self.num_factors
            ),
        )
        self.scorer = LinearProjectionLayer(**kwargs)