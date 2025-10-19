import torch.optim.lr_scheduler as lr_scheduler


SCHEDULER_REGISTRY = {
    "cosine": lr_scheduler.CosineAnnealingLR,
    "cosine_restart": lr_scheduler.CosineAnnealingWarmRestarts,
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "linear": lr_scheduler.LinearLR,
}