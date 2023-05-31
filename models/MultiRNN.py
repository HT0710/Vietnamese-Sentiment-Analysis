from lightning_modules.model import LitModel
import torch.nn as nn


class MultiRNN(LitModel):
    """Multiple RNN-based layers"""

    def __init__(
            self,
            vocab_size: int,
            output_size: int = 1,
            embedding_size: int = 400,
            hidden_size: int = 128,
        ):
        super().__init__()
        gru_hs = 256
        config = {
            "dropout": 0.25,
            "num_layers": 10,
            "batch_first": True,
            "bidirectional": False,
        }
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn0 = nn.GRU(embedding_size, gru_hs, **config)
        self.norm = nn.LayerNorm(gru_hs)
        self.rnn = nn.GRU(gru_hs, gru_hs, **config)
        self.fc = nn.Sequential(
            nn.Linear(gru_hs, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.rnn0(out)
        out = self.norm(out)
        out, _ = self.rnn(out)
        out = self.norm(out)
        out, _ = self.rnn(out)
        out = self.norm(out)
        out, _ = self.rnn(out)
        out = self.norm(out)
        out, _ = self.rnn(out)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
