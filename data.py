import os, re
import requests
from typing import Tuple, Optional, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.functional import truncate, pad_sequence
from lightning.pytorch import LightningDataModule

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd


class DataPreparation():
    """Data Preparation"""

    def __init__(self, stem: bool=False, lemma: bool=False):
        self.stemmer = PorterStemmer() if stem else None
        self.lemmatizer = WordNetLemmatizer() if lemma else None
        try:
            stopwords.words('english')
        except:
            self.setup()
        self.stopwords = set(stopwords.words('english'))

    def setup(self):
        print("Download required packages...")
        nltk.download('stopwords')  # stopwords
        nltk.download('punkt')      # tokenize
        nltk.download('wordnet')    # stem & lemma

    def __call__(self, text: str):
        return self.auto(text)

    def auto(self, text: str):
        out = self.remove_link(text)
        out = self.remove_html(out)
        out = self.remove_special(out)
        out = self.remove_numbers(out)
        out = self.remove_non_ascii(out)
        out = self.remove_emoji(out)
        out = self.remove_repeated(out)
        out = self.tokenize(out)
        out = self.remove_stopwords(out)
        out = self.stemming(out)
        out = self.lemmatization(out)
        return ' '.join(out)

    def remove_link(self, text: str):
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, ' ', text)

    def remove_html(self, text: str):
        return re.sub(r'<[^>]+>', ' ', text)

    def remove_special(self, text: str):
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    def remove_numbers(self, text: str):
        return re.sub(r'\d+', ' ', text)

    def remove_non_ascii(self, text: str):
        return re.sub(r'[^\x00-\x7f]', ' ', text)

    def remove_emoji(self, text: str):
        emojis = re.compile(
            '['
            u'\U0001F600-\U0001F64F'
            u'\U0001F300-\U0001F5FF'
            u'\U0001F680-\U0001F6FF'
            u'\U0001F1E0-\U0001F1FF'
            u'\U00002702-\U000027B0'
            u'\U000024C2-\U0001F251'
            ']+',
            flags=re.UNICODE
        )
        return emojis.sub(' ', text)

    # heeelloo worlddddd -> hello world
    def remove_repeated(self, text: str):
        return re.sub(r'(.)\1+', r'\1\1', text)

    def tokenize(self, text: str):
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: list):
        return [word for word in tokens if word not in self.stopwords]

    def stemming(self, tokens: list):
        return [self.stemmer.stem(word) for word in tokens] if self.stemmer else tokens

    def lemmatization(self, tokens: list):
        return [self.lemmatizer.lemmatize(word) for word in tokens] if self.lemmatizer else tokens


class DataPreprocessing():
    """Data Preprocessing"""

    def __init__(
            self,
            vocab: list = None,
            seq_length: int = 128,
            min_freq: int|float = 1,
            max_freq: int|float = 1.,
        ):
        """Data preprocessing"""
        self.vocab = vocab
        self.seq_length = seq_length
        self.vocab_conf = {
            'min_freq': min_freq,
            'max_freq': max_freq,
        }

    def __call__(self, texts: list[str]):
        """Call class directly"""
        return self.auto(texts)

    def auto(self, corpus: list[str]):
        """Auto pass through all step"""
        print("Building vocabulary...", end='\r')
        vocab = self.build_vocabulary(corpus, **self.vocab_conf) if not self.vocab else self.vocab
        print("Converting word2int...", end='\r')
        encoded = self.word2int(corpus, vocab)
        print("Truncating...         ", end='\r')
        truncated = self.truncate_sequences(encoded, seq_length=self.seq_length)
        print("Padding...            ", end='\r')
        padded = self.pad_sequences(truncated)
        return padded

    def build_vocabulary(self, corpus: list[str], min_freq: int|float=1, max_freq: int|float=1.):
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

    def word2int(self, corpus: list[str], vocab: dict):
        """Convert words to integer base on vocab"""
        encoder = lambda token: vocab[token] if token in vocab else self.vocab['<unk>']
        return [[encoder(token) for token in seq.split()] for seq in corpus]

    def truncate_sequences(self, corpus: list[str], seq_length=128):
        """Truncate the sequences"""
        return truncate(corpus, seq_length)

    def pad_sequences(self, corpus: list[str]):
        """Padding all sequences with the length equal to the longest sequence"""
        to_tensor = [torch.as_tensor(seq) for seq in corpus]
        return pad_sequence(to_tensor, batch_first=True)


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
        label = torch.as_tensor(
            label, dtype=torch.float32
        ).unsqueeze(0)
        return text, label


class IMDBDataModule(LightningDataModule):
    """IMDB Lightning Data Module"""

    def __init__(
            self,
            data_path: str = 'datasets/IMDB.csv',
            download: bool = True,
            preprocessing: Any = DataPreprocessing(),
            train_val_test_split: Tuple = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    @property
    def num_classes(self):
        return 2

    @property
    def vocab_size(self):
        preprocesser = self.hparams.preprocessing
        if not preprocesser.vocab:
            corpus = pd.read_csv(self.hparams.data_path)['text'].tolist()
            vocab = preprocesser.build_vocabulary(corpus, **preprocesser.vocab_conf)
        else:
            vocab = preprocesser.vocab
        return len(vocab)

    # Download the dataset
    def _download_data(self):
        data_path = 'datasets/IMDB.csv'
        url = 'https://raw.githubusercontent.com/HT0710/Sentiment-Analysis/data/IMDB_processed.csv'
        response = requests.get(url)
        if response.status_code != 200:
            print("Error occurred while downloading the file."); exit()
        else:
            print(f"Start download into '{data_path}'")
            os.mkdir('datasets') if not os.path.exists('datasets') else None
            with open(data_path, "wb") as file:
                file.write(response.content)
            print("File downloaded successfully.")
            return data_path

    def prepare_data(self):
        if not os.path.exists(self.hparams.data_path):
            print("Dataset not found!")
            self.hparams.data_path = self._download_data() if self.hparams.download else exit()
        dataset = pd.read_csv(self.hparams.data_path)
        self.labels = dataset['label'].tolist()
        raw_corpus = dataset['text'].tolist()
        self.corpus = self.hparams.preprocessing(raw_corpus)

    def setup(self, stage: str):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = DataModule(self.corpus, self.labels)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
            )

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.dl_conf, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.dl_conf, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.dl_conf, shuffle=False)
