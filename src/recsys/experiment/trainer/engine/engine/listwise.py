from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ListwiseEngine:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler(device=DEVICE)
        self.current_epoch = 0

    def __call__(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
    ):
        self.current_epoch += 1

        kwargs = dict(
            dataloader=trn_loader,
            obj="TRN",
        )
        self.model.train()
        trn_loss = self.epoch_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            obj="VAL",
        )
        with torch.no_grad():
            self.model.eval()
            val_loss = self.epoch_step(**kwargs)

        return trn_loss, val_loss

    def epoch_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        obj: str,
    ):
        epoch_loss = 0.0

        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {self.current_epoch} {obj}"
        )
        for user_idx, pos_idx, neg_idx in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                pos_idx=pos_idx.to(DEVICE), 
                neg_idx=neg_idx.to(DEVICE),
                obj=obj,
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_loss = self.batch_step(**kwargs)

            # backward pass
            if obj=="TRN":
                self.run_backprop(batch_loss)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def batch_step(self, user_idx, pos_idx, neg_idx, obj):
        fn = (
            self.model.estimate
            if obj=="TRN"
            else self.model.predict
        )
        pos_logit = fn(
            user_idx=user_idx, 
            item_idx=pos_idx,
        )
        neg_logit = fn(
            user_idx=user_idx.unsqueeze(1).expand_as(neg_idx).reshape(-1),
            item_idx=neg_idx.reshape(-1),
        ).view_as(neg_idx)
        loss = self.criterion(
            pos=pos_logit, 
            neg=neg_logit,
        )
        return loss

    def run_backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()