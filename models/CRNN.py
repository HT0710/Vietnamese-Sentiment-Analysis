from lightning_modules import LitModel
import torch.nn as nn


class CRNN(LitModel):
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
        self.features = self._construction(
            in_channels=embedding_size,
            config=[64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        )
        self.gru = nn.GRU(37, hidden_size, **config)
        self.lnorm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            in_features=hidden_size*2 if config['bidirectional'] else hidden_size,
            out_features=output_size
        )
        self.sigmoid = nn.Sigmoid()

    def _construction(self, in_channels, config):
        sequence = nn.Sequential()
        for x in config:
            if x == 'M':
                sequence.extend([nn.MaxPool1d(kernel_size=2, stride=2)])
            else:
                sequence.extend([
                    nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm1d(x),
                    nn.ReLU(True)
                ])
                in_channels = x
        return sequence

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)
        out = self.features(out)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.lnorm(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
