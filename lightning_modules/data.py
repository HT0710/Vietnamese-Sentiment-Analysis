import os, re, requests
from typing import Any, Tuple, Union, List, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.functional import truncate, pad_sequence
from lightning.pytorch import LightningDataModule
from transformers import AutoTokenizer

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import py_vncorenlp as vncore
import underthesea as uts
from pyvi import ViUtils

import pandas as pd
from rich import print


class DataPreprocesser():
    """Base class: Data Preparation"""

    def __init__(self, stopwords: bool=True, uncased: bool=True, char_limit: int=10, number_limit: int=-1):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = []
        self._config = {
            'stopwords': stopwords,
            'uncased': uncased,
            'number_limit': number_limit,
            'char_limit': char_limit
        }

    def __call__(self, text: str):
        return self.auto(text)

    def remove_link(self, text: str):
        """Remove website link. Example: https://..."""
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, ' ', text)

    def remove_html(self, text: str):
        """Remove html tag. Example: <abc>...</abc>"""
        return re.sub(r'<[^>]+>', ' ', text)

    def remove_punctuation(self, text: str):
        """Remove punctuation. Exmaple: !"#$%&'()*+,..."""
        return re.sub(r'[^\w\s]', ' ', text)

    def remove_non_ascii(self, text: str):
        """Remove non-ascii charactors"""
        return re.sub(r'[^\x00-\x7f]', ' ', text)

    def remove_emoji(self, text: str):
        """Remove emoji"""
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

    def remove_repeated(self, text: str):
        """Remove repeated charactor
        Example: heeelloo worlddddd -> hello world
        """
        return re.sub(r'(.)\1+', r'\1\1', text)

    def uncased(self, text: str):
        """Remove capitalization. Example: AbCDe -> abcde"""
        return text.lower()

    def tokenize(self, text: str):
        """Word tokenize. Example: hello world -> ["hello", "world"]"""
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]):
        """Remove stopwords. Example: a, an, the, this, that, ..."""
        return [word for word in tokens if word not in self.stopwords]

    def remove_incorrect(self, tokens: List[str], min_length: int=0, max_length: int=10):
        """Remove incorrect word.
        Remove words have length longer than max_length
        Example: with max_length=3 then ["1", "22", "333", "4444", "55555"] -> ["1", "22", "333"]
        """
        check = lambda x: True if (min_length <= len(x) <= max_length) else ('_' in x)
        return [word for word in tokens if check(word)]

    def format_numbers(self, tokens: List[str], max: int=100):
        """Replace number with token '<num>' if it is greater than max
        Example: with max=5 then ["2", "abc", "125", "69"] -> ["2", "abc", "<num>", "<num>"]
        """
        def check_number(x: str):
            if not x.isdigit():
                return x
            return '<num>' if int(x) > max else x
        return [check_number(word) for word in tokens]

    def remove_duplicated(self, tokens: List[str]):
        """Remove duplicated words
        Words appear consecutively more than 2 times
        Example: 1 22 333 4444 -> 1 22 33 44
        """
        tokens.extend([0, 1])
        return [a for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]) if not (a == b == c)]

    def stemming(self, tokens: List[str]):
        """Apply stemming algorithm"""
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatization(self, tokens: List[str]):
        """Apply lemmatization algorithm"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def auto(self, text: str, string: bool=True) -> Union[str, List[str]]:
        """Auto apply all of the methods.
        :Param string: return `str` if `true` else `list`
        """
        out = self.remove_link(text)
        out = self.remove_html(out)
        out = self.remove_punctuation(out)
        out = self.remove_non_ascii(out)
        out = self.remove_emoji(out)
        out = self.remove_repeated(out)
        out = self.uncased(out) if self._config['uncased'] else out
        out = self.tokenize(out)
        out = self.remove_stopwords(out) if not self._config['stopwords'] else out
        out = self.remove_incorrect(out, min_length=0, max_length=self._config['char_limit'])
        out = self.format_numbers(out, max=self._config['number_limit'])
        out = self.remove_duplicated(out)
        return ' '.join(out) if string else out


class EnPreprocesser(DataPreprocesser):
    """English Data Preparation"""

    def __init__(
            self,
            stopwords: bool = True,
            uncased: bool = True,
            char_limit: int = 10,
            number_limit: int = -1,
            stem: bool = False,
            lemma: bool = False
        ):
        super().__init__(stopwords, uncased, char_limit, number_limit)
        self._setup()
        self.stopwords = self._get_stopwords()
        self.config = {'stem': stem, 'lemma': lemma}

    def _setup(self):
        requirements = ['punkt', 'stopwords', 'wordnet']
        [nltk.download(r, quiet=True) for r in requirements]

    def _get_stopwords(self):
        return set(stopwords.words('english'))

    def auto(self, text: str):
        """Auto apply for english dataset"""
        out = super().auto(text, string=False)
        out = self.stemming(out) if self.config['stem'] else out
        out = self.lemmatization(out) if self.config['lemma'] else out
        return ' '.join(out)


class VnPreprocesser(DataPreprocesser):
    """Vietnamese Data Preparation"""

    def __init__(
            self,
            tokenize: bool = True,
            stopwords: bool = True,
            uncased: bool = True,
            accents: bool = True,
            char_limit: int = 10,
            number_limit: int = -1
        ):
        super().__init__(stopwords, uncased, char_limit, number_limit)
        self.stopwords = self._get_stopwords()
        self.tokenizer = self._get_tokenizer()
        self.config = {'tokenize': tokenize, 'accents': accents}

    def _get_stopwords(self):
        url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
        responds = requests.get(url)
        stopwords = responds.text.split('\n')
        return set(stopwords)

    def _get_tokenizer(self):
        vncore_path = os.getcwd() + '/models/vncorenlp'
        if not os.path.exists(vncore_path):
            os.mkdir(vncore_path)
            vncore.download_model(save_dir=vncore_path)
        return vncore.VnCoreNLP(annotators=["wseg"], save_dir=vncore_path)

    def remove_non_ascii(self, text: str):
        return re.sub(r'ˋ', '', text)

    def tokenize(self, text: str):
        tokens = self.tokenizer.word_segment(text)
        return tokens[0].split(" ") if tokens else ['']

    def text_normalize(self, tokens: List[str]):
        return [uts.text_normalize(word) for word in tokens]

    def remove_accents(self, tokens: List[str]):
        """Remove words accents for Vietnamese dataset
        Example: hôm nay trời đẹp -> hom nay troi dep
        """
        return [str(ViUtils.remove_accents(word), "UTF-8") for word in tokens]

    def auto(self, text: str):
        """Auto apply for vietnamese dataset"""
        out = super().auto(text, string=False)
        out = self.remove_accents(out) if not self.config['accents'] else out
        return ' '.join(out)


class CustomEncoder():
    """Custom Encoder"""

    def __init__(self):
        raise EnvironmentError(
            "CustomEncoder is designed to be instantiated, "
            "use the `CustomEncoder.load(...)` method instead."
        )

    @classmethod
    def load(
            self,
            model_name: str = None,
            seq_length: int = None,
            min_freq: Union[int, float] = 1,
            max_freq: Union[int, float] = 1.,
        ):
        r"""
        Return: `BasicEncoder` if model_name is None else `TransformerEncoder`
        """
        if not model_name:
            return BasicEncoder(seq_length, min_freq, max_freq)
        else:
            return TransformerEncoder(model_name, max_length=seq_length)


class BasicEncoder():
    """Custom Encoder"""

    def __init__(
            self,
            seq_length: int = None,
            min_freq: Union[int, float] = 1,
            max_freq: Union[int, float] = 1.,
        ):
        """Data preprocessing"""
        self.seq_length = seq_length
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
        vocab = self.build_vocabulary(corpus, **self.vocab_conf)
        print("[bold]Preprocessing:[/] Converting word2int...", end='\r')
        encoded = self.word2int(corpus, vocab)
        print("[bold]Preprocessing:[/] Truncating...         ", end='\r')
        truncated = self.truncate_sequences(encoded, seq_length=self.seq_length)
        print("[bold]Preprocessing:[/] Padding...            ", end='\r')
        padded = self.pad_sequences(truncated)
        print("[bold]Preprocessing:[/] Done      ")
        return padded

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
        return {token: idx for idx, token in enumerate(token_list)}

    def word2int(self, corpus: List[str], vocab: Dict[str, int]):
        """Convert words to integer base on vocab"""
        convert = lambda token: vocab[token] if token in vocab else self.vocab['<unk>']
        return [[convert(token) for token in seq.split()] for seq in corpus]

    def truncate_sequences(self, corpus: List[str], seq_length: int):
        """Truncate the sequences"""
        if not seq_length:
            seq_length = max(corpus, key=lambda x: len(x.split(" ")))
        return truncate(corpus, seq_length)

    def pad_sequences(self, corpus: List[str]):
        """
        Padding to all sequences with the length equal to the longest sequence.

        Note: Padding exercute after truncated.
        """
        to_tensor = [torch.as_tensor(seq) for seq in corpus]
        return pad_sequence(to_tensor, batch_first=True)


class TransformerEncoder():
    """Transormer encoder for Hugging face model"""

    def __init__(
            self,
            model_name: str,
            padding: bool = True,
            truncation: bool = True,
            max_length: int = None,
        ):
        r"""
        Auto load the given model

        Params:
            - model_name: name of the pretrained model from hugging face
            - max_length: limit amount of worlds
            - padding: padding to the sequence after truncation or if it is shorter than max_length
            - truncation: truncate the sequence if it exceeds the max_length
        """
        self.model_name = model_name
        self.model = AutoTokenizer.from_pretrained(model_name)
        self.config = {
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "return_tensors": "pt",
            "verbose": False,
        }

    @property
    def vocab(self):
        """Return vocab"""
        return self.model.get_vocab()

    def __call__(self, corpus: List[str]):
        return self.encode(corpus)

    def encode(self, corpus: List[str]):
        """Encode the corpus"""
        out = self.model(corpus, **self.config)
        return out.input_ids


class CustomDataModule(LightningDataModule):
    """Custom Data Module for Lightning"""

    def __init__(
            self,
            data_path: str,
            encoder: Any,
            data_limit: Union[int, float] = None,
            train_val_test_split: Tuple[float, float, float] = (0.75, 0.1, 0.15),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.dataset = self._load_data(data_path, data_limit)
        self.split_size = train_val_test_split
        self.encoder = encoder
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    @property
    def classes(self):
        """Return all classes (labels)"""
        return sorted(set(self.dataset['label']))

    @property
    def num_classes(self):
        """Return number of classes (labels)"""
        return len(self.classes)

    @property
    def vocab_size(self):
        """Return number of vocab"""
        if not self.encoder:
            return None
        if not hasattr(self.encoder, "vocab"):
            corpus = self.dataset['text'].to_list()
            self.encoder.vocab = self.encoder.build_vocabulary(corpus, **self.encoder.vocab_conf)
        return len(self.encoder.vocab)

    def _load_data(self, path: str, limit: Union[int, float]=None):
        dataset = pd.read_csv(path).dropna()
        if limit > len(dataset):
            raise ValueError(
                "The dataset limit value must be smaller than the dataset length "
                "or between 0 and 1 if it is a float."
            )
        if 0 < limit < 1:
            limit = int(len(dataset)*limit)
        return dataset[:limit]

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
