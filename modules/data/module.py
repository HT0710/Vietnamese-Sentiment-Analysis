from typing import Tuple, Union, List
from rich import print
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch import LightningDataModule
from .encoder import BasicEncoder, TransformerEncoder



class DataModule(Dataset):
    """Pytorch Data Module"""

    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]
        return text, label


class CustomDataModule(LightningDataModule):
    """Custom Data Module for Lightning"""

    def __init__(
            self,
            data_path: str,
            data_limit: Union[int, float] = None,
            batch_size: int = 32,
            tokenizer: Union[str, bool] = None,
            max_length: int = None,
            min_freq: Union[int, float] = 1,
            max_freq: Union[int, float] = 1.,
            train_val_test_split: Tuple[float, float, float] = (0.75, 0.1, 0.15),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.dataset = self._load_data(data_path, data_limit)
        self.split_size = train_val_test_split
        self.encoder = self._get_encoder(tokenizer, max_length, min_freq, max_freq)
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    @property
    def classes(self):
        """Return classes (labels)"""
        return sorted(set(self.dataset['label']))

    @property
    def num_classes(self):
        """Return number of classes (labels)"""
        return len(self.classes)

    @property
    def vocab(self):
        """Return vocab"""
        if not self.encoder:
            return None
        if not hasattr(self.encoder, "vocab"):
            corpus = self.dataset['text'].to_list()
            self.encoder.build_vocabulary(corpus, **self.encoder.vocab_conf)
        return self.encoder.vocab

    @property
    def vocab_size(self):
        """Return number of vocab"""
        return len(self.vocab) if self.vocab else 0

    def _load_data(self, path: str, limit: Union[int, float]=None):
        dataset = pd.read_csv(path).dropna()
        if not limit:
            return dataset
        if limit > len(dataset):
            raise ValueError(
                "The dataset limit value must be smaller than the dataset length "
                "or between 0 and 1 if it is a float."
            )
        if 0 < limit < 1:
            limit = int(len(dataset)*limit)
        return dataset[:limit]

    def _get_encoder(
            self,
            tokenizer: Union[str, bool] = None,
            max_length: int = None,
            min_freq: Union[int, float] = 1,
            max_freq: Union[int, float] = 1.,
        ):
        if isinstance(tokenizer, str):
            return TransformerEncoder(tokenizer, max_length)
        elif tokenizer is None:
            return BasicEncoder(max_length, min_freq, max_freq)
        else:
            return None

    def encode_corpus(self, corpus: List[str]):
        """Corpus encoding"""
        print("[bold]Prepare data:[/] Encode corpus...", end='\r')
        if not self.encoder:
            tokens = [[int(token) for token in seq.split(",")] for seq in corpus]
            return torch.as_tensor(tokens, dtype=torch.long)
        else:
            return self.encoder(corpus)

    def encode_label(self, labels: List[str]):
        """Label encoding"""
        print("Encode label...", end='\r')
        distinct = {key: index for index, key in enumerate(self.classes)}
        tensor_labels = torch.as_tensor([distinct[x] for x in labels], dtype=torch.float)
        return tensor_labels.unsqueeze(1)

    def prepare_data(self):
        print("[bold]Prepare data:[/]", end='\r')
        if not hasattr(self, "corpus"):
            raw_corpus = self.dataset['text'].to_list()
            raw_labels = self.dataset['label'].to_list()
            self.corpus = self.encode_corpus(raw_corpus)
            self.labels = self.encode_label(raw_labels)
        print("[bold]Prepare data:[/] Done            ")

    def setup(self, stage: str):
        if not hasattr(self, "data_train"):
            dataset = DataModule(self.corpus, self.labels)
            self.data_train, self.data_val, self.data_test = random_split(dataset=dataset, lengths=self.split_size)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.dl_conf, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.dl_conf, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.dl_conf, shuffle=False)
