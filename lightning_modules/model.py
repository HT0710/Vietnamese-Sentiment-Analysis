from torchmetrics.functional import accuracy, f1_score
from lightning.pytorch import LightningModule, Trainer


class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(self):
        super().__init__()
        self.log_conf = {
            "on_step": False, "on_epoch": True, "prog_bar": False, "logger": True
        }

    def criterion(self, y_hat, y):
        raise Exception("You need to overwrite 'criterion' function")

    def configure_optimizers(self):
        raise Exception("You need to overwrite 'configure_optimizers' function")

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict(
            {"train/loss": loss, "train/accuracy": acc, "train/f1_score": f1},
            **self.log_conf
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict(
            {"val/loss": loss, "val/accuracy": acc, "val/f1_score": f1},
            **self.log_conf
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict(
            {"test/loss": loss, "test/accuracy": acc, "test/f1_score": f1},
            **self.log_conf
        )
