import os, re, requests
from typing import Tuple, Optional, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.functional import truncate, pad_sequence
from lightning.pytorch import LightningDataModule

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import underthesea as uts
from pyvi import ViUtils

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from rich.progress import track
from rich import print


class DataPreparation():
    """Data Preparation"""

    def __init__(self, lang: str ='en', config: dict=None):
        self.lang = self._check_language(lang)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = self._get_stopwords()
        self.config = config

    def _check_language(self, lang: str):
        lang = lang.strip().lower()
        if lang not in ['en', 'vn']:
            raise ValueError("Only 'en' or 'vn' are supported.")
        return lang

    def _get_stopwords(self):
        if self.lang == 'en':
            nltk.download('stopwords', quiet=True)
            stopword_list = stopwords.words('english')
        if self.lang == 'vn':
            url = 'https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt'
            responds = requests.get(url)
            stopword_list = responds.text.split('\n')
        return set(stopword_list)

    def __call__(self, text: str):
        config = self.config if isinstance(self.config, dict) else {}
        return self.auto(text, **config)

    def auto(
            self,
            text: str,
            tokenize: bool = True,
            stopwords: bool = True,
            accents: bool = True,
            number_limit: int = 0,
            char_limit: int = 10,
            stem: bool = False,
            lemma: bool = False,
        ):
        out = self.remove_link(text)
        out = self.remove_html(out)
        out = self.remove_punctuation(out)
        out = self.remove_non_ascii(out)
        out = self.remove_emoji(out)
        out = self.remove_repeated(out)
        out = self.format_numbers(out, max=number_limit)
        out = self.text_normalize(out) if (self.lang == "vn") else out
        out = self.tokenize(out)
        out = self.remove_stopwords(out) if not stopwords else out
        out = self.remove_incorrect(out, min_length=0, max_length=char_limit)
        if self.lang == "vn":
            out = self.remove_accents(out) if not accents else out
            out = self.format_words(out) if tokenize else out
        if self.lang == "en":
            nltk.download('wordnet', quiet=True)
            out = self.stemming(out) if stem else out
            out = self.lemmatization(out) if lemma else out
        return ' '.join(out)

    def remove_link(self, text: str):
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, ' ', text)

    def remove_html(self, text: str):
        return re.sub(r'<[^>]+>', ' ', text)

    def remove_punctuation(self, text: str):
        return re.sub(r'[^\w\s]', ' ', text)

    def remove_non_ascii(self, text: str):
        if self.lang == 'en':
            return re.sub(r'[^\x00-\x7f]', ' ', text)
        if self.lang == 'vn':
            return re.sub(r'ˋ', '', text)

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

    def format_numbers(self, text: str, max: int=100):
        check_num = lambda x: '<num>' if (int(x.group(0)) >= max) else x.group(0)
        return re.sub(r'\d+', check_num, text)

    def text_normalize(self, text: str):
        text = uts.text_normalize(text)
        return text.lower().strip()

    def tokenize(self, text: str):
        if self.lang == 'en':
            nltk.download('punkt', quiet=True)
            return nltk.word_tokenize(text)
        if self.lang == 'vn':
            return uts.word_tokenize(text, fixed_words=['<num>'])

    def remove_stopwords(self, tokens: list):
        return [word for word in tokens if word not in self.stopwords]

    def remove_incorrect(self, tokens: list, min_length: int=0, max_length: int=10):
        check = lambda x: True if (min_length <= len(x) <= max_length) else (" " in x)
        return [word for word in tokens if check(word)]

    def remove_accents(self, tokens: list):
        return [str(ViUtils.remove_accents(word), "UTF-8") for word in tokens]

    def format_words(self, tokens: list):
        return [word.replace(" ", "_") for word in tokens]

    def stemming(self, tokens: list):
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatization(self, tokens: list):
        return [self.lemmatizer.lemmatize(word) for word in tokens]


class DataPreprocessing():
    """Data Preprocessing"""

    def __init__(
            self,
            seq_length: int = 128,
            min_freq: int|float = 1,
            max_freq: int|float = 1.,
        ):
        """Data preprocessing"""
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
        print("[bold]Preprocessing:[/] Building vocabulary...", end='\r')
        vocab = self.build_vocabulary(corpus, **self.vocab_conf)
        print("[bold]Preprocessing:[/] Converting word2int...", end='\r')
        encoded = self.word2int(corpus, vocab)
        print("[bold]Preprocessing:[/] Truncating...         ", end='\r')
        truncated = self.truncate_sequences(encoded, seq_length=self.seq_length)
        print("[bold]Preprocessing:[/] Padding...            ", end='\r')
        padded = self.pad_sequences(truncated)
        print("[bold]Preprocessing:[/] Done      ")
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
        convert = lambda token: vocab[token] if token in vocab else self.vocab['<unk>']
        return [[convert(token) for token in seq.split()] for seq in corpus]

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
        text_tensor = torch.as_tensor(text, dtype=torch.long)
        label_tensor = torch.as_tensor(label, dtype=torch.float)
        return text_tensor, label_tensor


class CustomDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            url: str = None,
            preprocessing: Any = DataPreprocessing(),
            train_val_test_split: Tuple = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.dataset = self._load_data(data_path, url)
        self.preprocess = preprocessing
        self.split_size = train_val_test_split
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    @property
    def classes(self):
        raise NotImplementedError("Value is still not implemented in this subclass.")

    @property
    def num_classes(self):
        raise NotImplementedError("Value is still not implemented in this subclass.")

    @property
    def vocab_size(self):
        if not hasattr(self.preprocess, "vocab"):
            corpus = self.dataset['text'].tolist()
            self.preprocess.vocab = self.preprocess.build_vocabulary(corpus, **self.preprocess.vocab_conf)
        return len(self.preprocess.vocab)

    def _download_data(self, data_path: str, url: str):
        if url is None:
            raise NotImplementedError("URL was not set when trying to download.")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            os.mkdir('datasets') if not os.path.exists('datasets') else None
            with open(data_path, "wb") as file:
                for chunk in track(response.iter_content(chunk_size=4096), 'Downloading'):
                    file.write(chunk)

    def _load_data(self, data_path: str, url: str=None):
        if not os.path.exists(data_path):
            print(f"Dataset not found. Download to [bold][green]{data_path}[/][/]")
            self._download_data(data_path, url)
        return pd.read_csv(data_path).dropna()

    def _label_encoding(self, labels):
        encoder = OneHotEncoder()
        return encoder.fit_transform(labels).toarray()

    def prepare_data(self):
        raw_corpus = self.dataset['text'].array
        raw_labels = self.dataset['label'].array.reshape(-1, 1)
        self.corpus = self.preprocess(raw_corpus)
        self.labels = self._label_encoding(raw_labels)

    def setup(self, stage: str):
        if not (self.data_train and self.data_val and self.data_test):
            dataset = DataModule(self.corpus, self.labels)
            self.data_train, self.data_val, self.data_test = random_split(dataset=dataset, lengths=self.split_size)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.dl_conf, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.dl_conf, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.dl_conf, shuffle=False)


class IMDBDataModule(CustomDataModule):
    def __init__(
            self,
            preprocessing: Any = DataPreprocessing(),
            train_val_test_split: Tuple = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True,
    ):
        kwargs = locals().copy()
        [ kwargs.pop(x) for x in ['self', '__class__'] ]
        super().__init__(
            data_path='datasets/IMDB.csv',
            url='https://raw.githubusercontent.com/HT0710/Sentiment-Analysis/data/en/IMDB.csv',
            **kwargs
        )

    @property
    def classes(self):
        return ['Negative', 'Positive']

    @property
    def num_classes(self):
        return 2


class VietDataModule(CustomDataModule):
    def __init__(
            self,
            preprocessing: Any = DataPreprocessing(),
            train_val_test_split: Tuple = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True,
    ):
        kwargs = locals().copy()
        [ kwargs.pop(x) for x in ['self', '__class__'] ]
        super().__init__(
            data_path='datasets/viet_full.csv',
            url='https://raw.githubusercontent.com/HT0710/Sentiment-Analysis/data/vn/viet_full.csv',
            **kwargs
        )

    @property
    def classes(self):
        return ['Negative', 'Neutral', 'Positive']

    @property
    def num_classes(self):
        return 3
