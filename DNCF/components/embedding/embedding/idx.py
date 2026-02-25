import torch
import torch.nn as nn


class IDXEmbedding(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
    ):
        super().__init__()

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb_slice = self.embed_user(user_idx)
        item_emb_slice = self.embed_item(item_idx)
        return user_emb_slice, item_emb_slice

    def _set_up_components(self):
        self._create_embeddings()
        self._init_embeddings()

    def _create_embeddings(self):
        kwargs = dict(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_users,
        )
        self.embed_user = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_items,
        )
        self.embed_item = nn.Embedding(**kwargs)

    def _init_embeddings(self):
        kwargs = dict(
            tensor=self.embed_user.weight, 
            mean=0.0, 
            std=0.01,
        )
        nn.init.normal_(**kwargs)

        kwargs = dict(
            tensor=self.embed_item.weight, 
            mean=0.0, 
            std=0.01,
        )
        nn.init.normal_(**kwargs)