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
        self._init_layers()

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
        rep_user, rep_item = self.rep(user_idx, item_idx)
        pred_vector = rep_user * rep_item
        return pred_vector

    def rep(self, user_idx, item_idx):
        user_id = self.user_embed_id(user_idx)
        user_hist = self.user_embed_hist(user_idx, item_idx)
        rep_user = user_id + user_hist

        item_id = self.item_embed_id(item_idx)
        item_hist = self.item_embed_hist(user_idx, item_idx)
        rep_item = item_id + item_hist

        return rep_user, rep_item

    def user_embed_hist(self, user_idx, item_idx):
        # get user vector from interactions
        user_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_batch = torch.arange(user_idx.size(0))
        user_slice[user_batch, item_idx] = 0
        
        # projection
        user_sum = user_slice.sum(dim=1, keepdim=True)
        user_sum = torch.clamp(user_sum, min=1.0)
        proj_user = self.proj_u(user_slice.float()) / torch.sqrt(user_sum)

        return proj_user

    def item_embed_hist(self, user_idx, item_idx):
        # get item vector from interactions
        item_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_batch = torch.arange(item_idx.size(0))
        item_slice[item_batch, user_idx] = 0
        
        # projection
        item_sum = item_slice.sum(dim=1, keepdim=True)
        item_sum = torch.clamp(item_sum, min=1.0)
        proj_item = self.proj_i(item_slice.float()) / torch.sqrt(item_sum)

        return proj_item

    def _init_layers(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_embed_id = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed_id = nn.Embedding(**kwargs)

        nn.init.normal_(self.user_embed_id.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed_id.weight, mean=0.0, std=0.01)

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