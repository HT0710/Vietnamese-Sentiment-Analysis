from lightning_modules import LitModel
import torch.nn as nn
import torch


class RNNBlock(nn.Module):
    def __init__(self, input_size, output_size, num_layers, bidirectional):
        super().__init__()
        config = {
            "dropout": 0.25,
            "num_layers": num_layers,
            "batch_first": True,
            "bidirectional": bidirectional,
        }
        self.gru = nn.GRU(input_size, output_size, **config)
        self.norm = nn.LayerNorm(output_size * (config['bidirectional'] + 1))

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.norm(out)
        out = out[:, -1, :]
        return out


class RNNParallel(LitModel):

    num_rnn_block = 2

    def __init__(
            self,
            vocab_size: int,
            output_size: int = 1,
            embedding_size: int = 400,
            hidden_size: int = 128
        ):
        super().__init__()
        config = {
            "input_size": embedding_size,
            "output_size": hidden_size,
            "bidirectional": True
        }
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn_near = RNNBlock(num_layers=2, **config)
        self.rnn_far = RNNBlock(num_layers=10, **config)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(
            in_features=hidden_size * self.num_rnn_block * (config['bidirectional'] + 1),
            out_features=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        out_near = self.rnn_near(out)
        out_far = self.rnn_far(out)
        out = torch.cat((out_near, out_far), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
