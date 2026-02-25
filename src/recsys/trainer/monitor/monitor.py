import torch
import copy
from .predictor import Predictor
from .calculator import Calculator
from .early_stop import EarlyStopping


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monitor(object):
    def __init__(
        self,
        predictor: Predictor,
        calculator: Calculator,
        early_stop: EarlyStopping,
    ):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator
        self.early_stop = early_stop

    def __call__(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        kwargs = dict(
            dataloader=dataloader,
        )
        result = self.predictor(**kwargs)
        
        kwargs = dict(
            result=result,
        )
        score = self.calculator(**kwargs)

        kwargs = dict(
            current_score=score, 
            current_state=self.state,
        )
        self.early_stop(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.early_stop.should_stop

    @property
    def counter(self):
        return self.early_stop.counter

    @property
    def best_epoch(self):
        return (
            self.early_stop.best_epoch
            if self.early_stop.best_epoch!=0
            else self.early_stop.current_epoch
        )

    @property
    def best_score(self):
        return (
            self.early_stop.best_score
            if self.early_stop.best_score is not None
            else self.early_stop.current_score
        )

    @property
    def best_state(self):
        return (
            self.early_stop.best_state
            if self.early_stop.best_state is not None
            else self.early_stop.current_state
        )

    @property
    def state(self):
        return copy.deepcopy(self.predictor.model.state_dict())