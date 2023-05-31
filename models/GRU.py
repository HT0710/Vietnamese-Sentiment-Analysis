from lightning_modules import LitModel
import torch.nn as nn


class GRU(LitModel):
    def __init__(
            self,
            vocab_size: int,
            output_size: int = 1,
            embedding_size: int = 400,
            hidden_size: int = 128
        ):
        super().__init__()
        config = {
            "dropout": 0.25,
            "num_layers": 2,
            "batch_first": True,
            "bidirectional": True,
        }
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, **config)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            in_features=hidden_size*2 if config['bidirectional'] else hidden_size,
            out_features=output_size
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
