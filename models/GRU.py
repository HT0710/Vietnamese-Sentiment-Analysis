from lightning_modules import LitModel
import torch.nn as nn


class GRU(LitModel):
    def __init__(
            self,
            vocab_size: int,
            output_size: int = 1,
            hidden_size: int = 128,
            embedding_size: int = 400,
            num_layers: int = 2,
            dropout: float = 0.2,
            batch_first: bool = True,
            bidirectional: bool = False,
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(
            input_size = embedding_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = batch_first,
            bidirectional = bidirectional
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            in_features = hidden_size*2 if bidirectional else hidden_size,
            out_features = output_size
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
