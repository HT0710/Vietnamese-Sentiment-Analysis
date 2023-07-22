from modules.lit_model import LitModel
from transformers import RobertaModel, GPT2Model
import torch.nn as nn



class Classification(LitModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class BERT(Classification):
    def __init__(self, model_name: str, output_size: int,  hidden_size: int=256, **kwargs):
        super().__init__(768, output_size, hidden_size)
        self.bert = RobertaModel.from_pretrained(model_name)

    def forward(self, x):
        out = self.bert(x).pooler_output
        out = super().forward(out)
        return out


class GPT2(Classification):
    def __init__(self, model_name: str, output_size: int,  hidden_size: int=256, **kwargs):
        super().__init__(768, output_size, hidden_size)
        self.gpt2 = GPT2Model.from_pretrained(model_name)

    def forward(self, x):
        out = self.gpt2(x).last_hidden_state
        out = super().forward(out).squeeze(2)
        out = out.max(dim=1)[0].unsqueeze(1)
        return out
