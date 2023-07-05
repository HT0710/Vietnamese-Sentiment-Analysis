from modules.lit_model import LitModel
import torch.nn as nn



class RNNBase(LitModel):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.model(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


config = {
    "num_layers": 2, "dropout": 0.3, "batch_first": True
}


class RNN(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.RNN(embedding_size, hidden_size, bidirectional=False, **config)

class LSTM(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.LSTM(embedding_size, hidden_size, bidirectional=False, **config)

class GRU(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.GRU(embedding_size, hidden_size, bidirectional=False, **config)


class BiRNN(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.RNN(embedding_size, hidden_size, bidirectional=True, **config)
        self.fc = nn.Linear(hidden_size*2, output_size)

class BiLSTM(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.LSTM(embedding_size, hidden_size, bidirectional=True, **config)
        self.fc = nn.Linear(hidden_size*2, output_size)

class BiGRU(RNNBase):
    def __init__(self, vocab_size: int, output_size: int, embedding_size: int=128, hidden_size: int=256, **kwagrs):
        super().__init__(vocab_size, output_size, embedding_size, hidden_size)
        self.model = nn.GRU(embedding_size, hidden_size, bidirectional=True, **config)
        self.fc = nn.Linear(hidden_size*2, output_size)
