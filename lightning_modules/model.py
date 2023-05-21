from torchmetrics.functional import accuracy, f1_score
from lightning.pytorch import LightningModule


class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(self):
        super().__init__()
        self.log_conf = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
        }
    
    def criterion(self, y_hat, y):
        raise Exception("Overwrite this to return a loss function")
    
    def configure_optimizers(self):
        raise Exception("Overwrite this to return an optimizer")

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("train/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"train/accuracy": acc, "train/f1_score": f1}, prog_bar=False, **self.log_conf,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("val/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"val/accuracy": acc, "val/f1_score": f1}, prog_bar=False, **self.log_conf,
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("test/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"test/accuracy": acc, "test/f1_score": f1}, prog_bar=False, **self.log_conf,
        )
