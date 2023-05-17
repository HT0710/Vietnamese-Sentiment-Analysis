import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn

from torchmetrics.functional import accuracy, f1_score
from lightning.pytorch import LightningModule


class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(self, model: nn.Module, lr: float=1e-3, save_hparams=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'], logger=False)
        self.model = model
        self.learning_rate = lr
        self.log_conf = {
            "on_step": False,
            "on_epoch": True,
            "logger": True,
        }

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("train/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"train/accuracy": acc, "train/f1_score": f1},
            prog_bar=False, **self.log_conf,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("val/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"val/accuracy": acc, "val/f1_score": f1},
            prog_bar=False, **self.log_conf,
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log("test/loss", loss, prog_bar=True, **self.log_conf)
        self.log_dict(
            {"test/accuracy": acc, "test/f1_score": f1},
            prog_bar=False, **self.log_conf,
        )
