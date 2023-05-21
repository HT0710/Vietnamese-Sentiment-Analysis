from lightning_modules.model import LitModel
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class GRU(LitModel):
    def __init__(
            self,
            vocab_size: int,
            output_size: int,
            hidden_size: int = 128,
            embedding_size: int = 400,
            n_layers: int = 2,
            dropout: float = 0.2,
            save_hparams = None
        ):
        super().__init__()
        self.save_hyperparameters('save_hparams')
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        out, _ =  self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def configure_optimizers(self):
        learning_rate = self.hparams.save_hparams['trainer']['learning_rate']
        return optim.Adam(self.parameters(), lr=learning_rate)
