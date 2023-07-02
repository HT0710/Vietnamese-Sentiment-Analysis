from lightning_modules import LitModel
from transformers import RobertaModel
import torch.nn as nn


class BERT(LitModel):
    def __init__(self, output_size: int = 1,  hidden_size: int = 256, **kwargs):
        super().__init__()
        self.embedding = RobertaModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x).pooler_output
        out = self.dropout(out)
        out = self.norm(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
