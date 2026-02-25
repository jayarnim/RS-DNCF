import torch
import torch.nn as nn


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(
    model: nn.Module, 
    path: str,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "init_args": model.init_args,
    }

    torch.save(checkpoint, path)


def load_model(
    model_cls: nn.Module, 
    path: str, 
):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = model_cls(**checkpoint["init_args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
