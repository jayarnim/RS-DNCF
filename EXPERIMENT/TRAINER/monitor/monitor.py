import copy
import torch
import torch.nn as nn


class Monitor:
    def __init__(
        self,
        model: nn.Module,
        predictor,
        metrics_computer,
        early_stopper,
    ):
        self.model = model
        self.predictor = predictor
        self.metrics_computer = metrics_computer
        self.early_stopper = early_stopper

    def __call__(
        self,
        dev_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
    ):
        kwargs = dict(
            dev_loader=dev_loader,
            epoch=epoch,
        )
        result = self.predictor(**kwargs)
        
        score = self.metrics_computer(result)

        kwargs = dict(
            current_score=score, 
            current_epoch=epoch,
            current_model_state=copy.deepcopy(self.model.state_dict()),
        )
        self.early_stopper(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.early_stopper.should_stop

    @property
    def counter(self):
        return self.early_stopper.counter

    @property
    def best_epoch(self):
        return self.early_stopper.best_epoch

    @property
    def best_score(self):
        return self.early_stopper.best_score

    @property
    def best_model_state(self):
        return self.early_stopper.best_model_state