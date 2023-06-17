import re, requests
from typing import Tuple, Any
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
from rich import print


class DataPreparation():
    """Data Preparation"""

    def __init__(self, stopwords: bool=True, char_limit: int=10, number_limit: int=-1):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = []
        self._config = {
            'stopwords': stopwords,
            'number_limit': number_limit,
            'char_limit': char_limit
        }

    def __call__(self, text: str):
        return self.auto(text)

    def remove_link(self, text: str):
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, ' ', text)

    def remove_html(self, text: str):
        return re.sub(r'<[^>]+>', ' ', text)

    def remove_punctuation(self, text: str):
        return re.sub(r'[^\w\s]', ' ', text)

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

    def text_normalize(self, text: str):
        return text.lower().strip()

    def tokenize(self, text: str):
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: list):
        return [word for word in tokens if word not in self.stopwords]

    def remove_incorrect(self, tokens: list, min_length: int=0, max_length: int=10):
        check = lambda x: True if (min_length <= len(x) <= max_length) else (' ' in x)
        return [word for word in tokens if check(word)]

    def format_numbers(self, tokens: list, max: int=100):
        """Replace number with token '<num>' if it is greater than max
        Example: if max=5 then ["2", "abc", "125", "69"] -> ["2", "abc", "<num>", "<num>"]
        """
        def check_number(x: str):
            if not x.isdigit():
                return x
            return '<num>' if int(x) > max else x
        return [check_number(word) for word in tokens]

    def remove_duplicated(self, tokens: list):
        """Remove duplicated words
        Words appear consecutively more than 2 times
        Example: 1 22 333 4444 -> 1 22 33 44
        """
        tokens.extend([0, 1])
        return [a for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]) if not (a == b == c)]

    def stemming(self, tokens: list):
        """Apply stemming algorithm"""
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatization(self, tokens: list):
        """Apply lemmatization algorithm"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def remove_accents(self, tokens: list):
        """Remove words accents for Vietnamese dataset
        Example: hôm nay trời đẹp -> hom nay troi dep
        """
        return [str(ViUtils.remove_accents(word), "UTF-8") for word in tokens]

    def format_words(self, tokens: list):
        """Connect tokenized words for Vietnamese dataset
        Example: ["hôm nay", "Hồ Chí Minh", "trời", "đẹp"] -> ["hôm_nay", "Hồ_Chí_Minh", "trời", "đẹp"]
        """
        return [word.replace(' ', '_') for word in tokens]

    def auto(self, text: str, string: bool=True) -> str|list:
        out = self.remove_link(text)
        out = self.remove_html(out)
        out = self.remove_punctuation(out)
        out = self.remove_non_ascii(out)
        out = self.remove_emoji(out)
        out = self.remove_repeated(out)
        out = self.text_normalize(out)
        out = self.tokenize(out)
        out = self.remove_stopwords(out) if not self._config['stopwords'] else out
        out = self.remove_incorrect(out, min_length=0, max_length=self._config['char_limit'])
        out = self.format_numbers(out, max=self._config['number_limit'])
        out = self.remove_duplicated(out)
        return ' '.join(out) if string else out


class EnPreparation(DataPreparation):
    def __init__(
            self,
            stopwords: bool = True,
            char_limit: int = 10,
            number_limit: int = -1,
            stem: bool = False,
            lemma: bool = False
        ):
        super().__init__(stopwords, char_limit, number_limit)
        self._setup()
        self.stopwords = self._get_stopwords()
        self.config = {'stem': stem, 'lemma': lemma}

    def _setup(self):
        requirements = ['punkt', 'stopwords', 'wordnet']
        [nltk.download(r, quiet=True) for r in requirements]

    def _get_stopwords(self):
        return set(stopwords.words('english'))

    def auto(self, text: str):
        out = super().auto(text, string=False)
        out = self.stemming(out) if self.config['stem'] else out
        out = self.lemmatization(out) if self.config['lemma'] else out
        return ' '.join(out)


class VnPreparation(DataPreparation):
    def __init__(
            self,
            tokenize: bool = True,
            stopwords: bool = True,
            accents: bool = True,
            char_limit: int = 10,
            number_limit: int = -1
        ):
        super().__init__(stopwords, char_limit, number_limit)
        self.stopwords = self._get_stopwords()
        self.config = {'tokenize': tokenize, 'accents': accents}

    def _get_stopwords(self):
        url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
        responds = requests.get(url)
        stopwords = responds.text.split('\n')
        return set(stopwords)

    def remove_non_ascii(self, text: str):
        return re.sub(r'ˋ', '', text)

    def text_normalize(self, text: str):
        text = uts.text_normalize(text)
        return text.lower().strip()

    def tokenize(self, text: str):
        return uts.word_tokenize(text)

    def auto(self, text: str):
        out = super().auto(text, string=False)
        out = self.remove_accents(out) if not self.config['accents'] else out
        out = self.format_words(out) if self.config['tokenize'] else out
        return ' '.join(out)


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
        return text, label


class CustomDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            preprocessing: Any = DataPreprocessing(),
            train_val_test_split: Tuple = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.dataset = pd.read_csv(data_path).dropna()
        self.preprocess = preprocessing
        self.split_size = train_val_test_split
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    @property
    def classes(self):
        return set(self.dataset['label'])

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def vocab_size(self):
        if not hasattr(self.preprocess, "vocab"):
            corpus = self.dataset['text'].values
            self.preprocess.vocab = self.preprocess.build_vocabulary(corpus, **self.preprocess.vocab_conf)
        return len(self.preprocess.vocab)

    def _label_encode(self, labels):
        distinct = {key: index for index, key in enumerate(sorted(set(labels)))}
        tensor_labels = torch.as_tensor([distinct[x] for x in labels], dtype=torch.float)
        return tensor_labels.unsqueeze(1)

    def prepare_data(self):
        if not hasattr(self, "corpus"):
            raw_corpus = self.dataset['text'].values
            raw_labels = self.dataset['label'].values
            self.corpus = self.preprocess(raw_corpus)
            self.labels = self._label_encode(raw_labels)

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
