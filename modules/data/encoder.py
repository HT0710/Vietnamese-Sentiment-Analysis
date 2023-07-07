from typing import Union, List, Dict
from collections import Counter
from rich import print

import torch
from torchtext.functional import truncate, pad_sequence
from transformers import AutoTokenizer



class BasicEncoder():
    """Basic Encoder"""

    def __init__(
            self,
            max_length: int = None,
            min_freq: Union[int, float] = 1,
            max_freq: Union[int, float] = 1.,
        ):
        """Basic encoder"""
        self.max_length = max_length
        self.vocab_conf = {
            'min_freq': min_freq,
            'max_freq': max_freq,
        }

    def __call__(self, texts: List[str]):
        """Call class directly"""
        return self.auto(texts)

    def auto(self, corpus: List[str]):
        """Auto pass through all step"""
        print("[bold]Preprocessing:[/] Building vocabulary...", end='\r')
        out = self.build_vocabulary(corpus, **self.vocab_conf)
        print("[bold]Preprocessing:[/] Converting word2int...", end='\r')
        out = self.word2int(corpus, out)
        print("[bold]Preprocessing:[/] Truncating...         ", end='\r')
        out = self.truncate_sequences(out, max_length=self.max_length)
        print("[bold]Preprocessing:[/] Padding...            ", end='\r')
        out = self.pad_sequences(out)
        print("[bold]Preprocessing:[/] Done      ")
        return out

    def encode(self, corpus: List[str]):
        if not hasattr(self, "vocab"):
            raise AttributeError("Vocab not found.")
        return torch.as_tensor(self.word2int(corpus, self.vocab))

    def build_vocabulary(self, corpus: List[str], min_freq: Union[int, float]=1, max_freq: Union[int, float]=1.):
        """Build vocabulary
        - Can be limited with min frequency and max frequence
        - If `int`: number of the token
        - If `float`: percent of the token
        """
        tokenized = ' '.join(corpus).split(' ')
        min_freq = int(min_freq * len(tokenized)) if isinstance(min_freq, float) else min_freq
        max_freq = int(max_freq * len(tokenized)) if isinstance(max_freq, float) else max_freq
        counter = Counter(tokenized)
        token_list = ['<pad>', '<unk>']
        token_list.extend([token for token in counter if (min_freq <= counter[token] <= max_freq)])
        self.vocab = {token: idx for idx, token in enumerate(token_list)}
        return self.vocab

    def word2int(self, corpus: List[str], vocab: Dict[str, int]):
        """Convert words to integer base on vocab"""
        convert = lambda token: vocab[token] if token in vocab else self.vocab['<unk>']
        return [[convert(token) for token in seq.split()] for seq in corpus]

    def truncate_sequences(self, corpus: List[str], max_length: int):
        """Truncate the sequences"""
        if not max_length:
            max_length = max(corpus, key=lambda x: len(x.split(" ")))
        return truncate(corpus, max_length)

    def pad_sequences(self, corpus: List[str]):
        """
        Padding to all sequences with the length equal to the longest sequence.

        Note: Padding exercute after truncated.
        """
        to_tensor = [torch.as_tensor(seq) for seq in corpus]
        return pad_sequence(to_tensor, batch_first=True)


class TransformerEncoder():
    """Transformer encoder for Hugging face tokenizer"""

    def __init__(
            self,
            tokenizer: str,
            max_length: int = None,
            truncation: bool = True,
            padding: bool = True,
        ):
        r"""
        Auto load the given tokenizer

        Params:
            - model_name: name of the pretrained tokenizer from hugging face
            - max_length: limit amount of worlds
            - padding: padding to the sequence after truncation or if it is shorter than max_length
            - truncation: truncate the sequence if it exceeds the max_length
        """
        self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.config = {
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "return_tensors": "pt",
            "verbose": False,
        }

    @property
    def vocab(self):
        """Return vocab"""
        return self.tokenizer.get_vocab()

    def __call__(self, corpus: List[str]):
        return self.encode(corpus)

    def encode(self, corpus: List[str]):
        """Encode the corpus"""
        out = self.tokenizer(corpus, **self.config)
        return out.input_ids
