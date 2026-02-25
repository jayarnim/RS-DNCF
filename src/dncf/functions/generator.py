import torch.nn as nn


def fc_block(
    input_dim, 
    hidden_dim, 
    dropout,
):
    IN_FEATRUES = input_dim
    
    for OUT_FEATURES in hidden_dim:
        yield nn.Sequential(
            nn.Linear(IN_FEATRUES, OUT_FEATURES),
            nn.LayerNorm(OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        IN_FEATRUES = OUT_FEATURES