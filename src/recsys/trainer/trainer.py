from IPython.display import clear_output
import torch


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self, 
        engine,
        monitor,
        num_epochs,
    ):
        super().__init__()
        self.engine = engine
        self.monitor = monitor
        self.num_epochs = num_epochs

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
    ):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
        )
        records = self.progressor(**kwargs)

        clear_output(wait=False)

        print(
            "VALIDATION",
            f"\tBEST SCORE: {self.monitor.best_score:.4f}",
            f"\tBEST EPOCH: {self.monitor.best_epoch}",
            sep="\n",
        )

        return records

    def progressor(self, trn_loader, val_loader):
        trn_log_list = []
        val_log_list = []

        for epoch in range(self.num_epochs):
            # RUN ==========
            trn_loss = self.engine(trn_loader)
            val_score = self.monitor(val_loader)

            # ACCUMULATE ==========
            trn_log_list.append(trn_loss)
            val_log_list.append(val_score)

            # EARLY STOPPING ==========
            if self.monitor.should_stop==True:
                break

            # PRINT ==========
            print(
                f"CURRENT TRN LOSS: {trn_loss:.4f}",
            )
            print(
                f"CURRENT VAL METRIC: {val_score:.4f}",
                f"BEST METRIC: {self.monitor.best_score:.4f}",
                f"COUNTER: {self.monitor.counter}",
                sep='\t',
            )

            # LOG RESET ==========
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        self.model.load_state_dict(self.monitor.best_state)

        return dict(
            trn=trn_log_list,
            val=val_log_list,
        )

    @property
    def model(self):
        return self.engine.model