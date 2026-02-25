import torch
import torch.nn as nn


class HistoryEmbedding(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        projection_dim: int,
    ):
        super().__init__()

        # global attr
        self.register_buffer("user_interactions", interactions.float())
        self.register_buffer("item_interactions", interactions.T.contiguous().float())
        self.num_users = num_users
        self.num_items = num_items
        self.projection_dim = projection_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        kwargs = dict(
            anchor_idx=user_idx,
            target_idx=item_idx,
            anchor="user",
        )
        user_emb_slice = self._embedding_generator(**kwargs)

        kwargs = dict(
            anchor_idx=item_idx,
            target_idx=user_idx,
            anchor="item",
        )
        item_emb_slice = self._embedding_generator(**kwargs)
        
        return user_emb_slice, item_emb_slice

    def _embedding_generator(self, anchor_idx, target_idx, anchor):
        # get anchor interaction vector from interaction matrix
        interaction_vec = self.interactions[anchor][anchor_idx, :-1].clone().float()

        # masking target
        batch_idx = torch.arange(anchor_idx.size(0))
        interaction_vec[batch_idx, target_idx] = 0.0

        # projection
        num_hist = interaction_vec.sum(dim=1, keepdim=True)
        num_hist = torch.clamp(num_hist, min=1.0)
        projection_vec = self.projection[anchor](interaction_vec) / torch.sqrt(num_hist)

        return projection_vec

    @property
    def interactions(self):
        return dict(
            user=self.user_interactions,
            item=self.item_interactions,
        )

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.num_items,
            out_features=self.projection_dim,
            bias=False,
        )
        projection_user = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.num_users,
            out_features=self.projection_dim,
            bias=False,
        )
        projection_item = nn.Linear(**kwargs)

        components = dict(
            user=projection_user,
            item=projection_item,
        )
        self.projection = nn.ModuleDict(components)