import torch


class DefaultSelector(object):
    def __init__(
        self, 
        interactions: torch.Tensor,
        **params,
    ):
        super().__init__()
        self.interactions = interactions
    
    def __call__(self, **kwargs):
        rows, cols = self.interactions.nonzero(as_tuple=True)

        hist_indices = [
            [] 
            for _ in range(len(self.interactions))
        ]
        
        for row, col in zip(rows.tolist(), cols.tolist()):
            hist_indices[row].append(col)

        hist_indices = [
            torch.tensor(indices, dtype=torch.long)
            for indices in hist_indices
        ]

        return hist_indices