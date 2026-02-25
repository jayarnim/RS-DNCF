import torch


class FreqSelector(object):
    def __init__(
        self,
        interactions: torch.Tensor,
        max_hist: int,
    ):
        super().__init__()
        self.interactions = interactions
        self.max_hist = max_hist

    def __call__(self, **kwargs):
        freq_target = self.interactions.sum(dim=0)

        # select top-k indices
        topk_indices = []
        for row in range(len(self.interactions)):
            hist_count = int(self.interactions[row].sum().item())
            
            # padding only
            if hist_count == 0:
                indices = torch.tensor([0], dtype=torch.long)
                topk_indices.append(indices.to(torch.long))
            
            # all
            elif hist_count <= self.max_hist:
                indices = self.interactions[row].nonzero(as_tuple=True)[0]
                topk_indices.append(indices.to(torch.long))
            
            # top-k selection
            else:
                hist_idx = self.interactions[row].nonzero(as_tuple=True)[0]
                scores = freq_target[hist_idx]
                vals, indices = torch.topk(scores, k=self.max_hist)
                topk_indices.append(indices.to(torch.long))

        return topk_indices