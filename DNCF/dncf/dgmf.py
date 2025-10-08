import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
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
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # generate layers
        self._set_up_components()

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
        pred_vector = self.gmf(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def gmf(self, user_idx, item_idx):
        user_embed_slice, item_embed_slice = self.rep(user_idx, item_idx)
        pred_vector = user_embed_slice * item_embed_slice
        return pred_vector

    def rep(self, user_idx, item_idx):
        user_embed_slice_id = self.user_id_embed(user_idx)
        user_embed_slice_hist = self.user_hist_embed_generator(user_idx, item_idx)
        user_embed_slice = user_embed_slice_id + user_embed_slice_hist

        item_embed_slice_id = self.item_id_embed(item_idx)
        item_embed_slice_hist = self.item_hist_embed_generator(user_idx, item_idx)
        item_embed_slice = item_embed_slice_id + item_embed_slice_hist

        return user_embed_slice, item_embed_slice

    def user_hist_embed_generator(self, user_idx, item_idx):
        # get user vector from interactions
        user_interaction_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_idx_batch = torch.arange(user_idx.size(0))
        user_interaction_slice[user_idx_batch, item_idx] = 0
        
        # projection
        n_hist = user_interaction_slice.sum(dim=1, keepdim=True)
        n_hist = torch.clamp(n_hist, min=1.0)
        user_proj_slice = self.proj_u(user_interaction_slice.float()) / torch.sqrt(n_hist)

        return user_proj_slice

    def item_hist_embed_generator(self, user_idx, item_idx):
        # get item vector from interactions
        item_interaction_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_idx_batch = torch.arange(item_idx.size(0))
        item_interaction_slice[item_idx_batch, user_idx] = 0
        
        # projection
        n_hist = item_interaction_slice.sum(dim=1, keepdim=True)
        n_hist = torch.clamp(n_hist, min=1.0)
        item_proj_slice = self.proj_i(item_interaction_slice.float()) / torch.sqrt(n_hist)

        return item_proj_slice

    def _set_up_components(self):
        self._create_embeddings()
        self._create_layers()

    def _create_embeddings(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_id_embed = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_id_embed = nn.Embedding(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.n_items,
            out_features=self.n_factors,
            bias=False,
        )
        self.proj_u = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.n_users,
            out_features=self.n_factors,
            bias=False,
        )
        self.proj_i = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.n_factors,
            out_features=1, 
        )
        self.logit_layer = nn.Linear(**kwargs)