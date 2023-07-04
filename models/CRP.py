from modules.lit_model import LitModel
import torch.nn as nn
import torch



class ConvolutionWay(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.features = self._construction(
            in_channels=embedding_size,
            config=[64, 'M', 128, 'M', 256, 'M']
        )

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
        x = x.permute(0, 2, 1)
        out = self.features(x)
        return out


class RecurrentWay(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        hidden_size = 512
        config = {
            "dropout": 0.25,
            "num_layers": 2,
            "batch_first": True,
            "bidirectional": False,
        }
        self.gru1 = nn.LSTM(embedding_size, hidden_size, **config)
        self.gru2 = nn.LSTM(hidden_size, hidden_size, **config)
        self.norm = nn.LayerNorm(hidden_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.norm(out)
        out, _ = self.gru2(out)
        out = self.norm(out)
        out = self.maxpool(out)
        out = out[:, -1, :]
        return out


class CRP(LitModel):
    """CNN and RNN-based Parallel"""

    # Parameter
    seq_length = 200
    num_conv_maxpool = 3
    both_way_out_shape = 256

    flatten_shape = both_way_out_shape*(int(seq_length/(2**num_conv_maxpool))+1)

    def __init__(
            self,
            vocab_size: int,
            output_size: int = 1,
            embedding_size: int = 400,
            hidden_size: int = 128,
            **kwargs
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.cway = ConvolutionWay(embedding_size)
        self.rway = RecurrentWay(embedding_size)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_shape, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        c_out = self.cway(out)
        r_out = self.rway(out).unsqueeze(2)
        out = torch.cat((c_out, r_out), dim=2)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
