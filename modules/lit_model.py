from torchmetrics.functional import accuracy, f1_score
from lightning.pytorch import LightningModule
import torch.nn.functional as F
import torch.optim as optim
from rich import print
import torch
import os



class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(self):
        super().__init__()

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.trainer['learning_rate'])

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict(
            {"train/loss": loss, "train/accuracy": acc, "train/f1_score": f1},
            on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict({"val/loss": loss, "val/accuracy": acc, "val/f1_score": f1})

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')
        f1 = f1_score(y_hat, y, task='binary')
        self.log_dict({"test/loss": loss, "test/accuracy": acc, "test/f1_score": f1})

    def save_hparams(self, config: dict):
        self.hparams.update(config)
        self.save_hyperparameters()

    def load(self, path: str, strict: bool=True, verbose: bool=True):
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_state_dict = torch.load(path, map_location=device)['state_dict']
        self.load_state_dict(checkpoint_state_dict, strict=strict)
        print("[bold]Load checkpoint successfully.[/]") if verbose else None
