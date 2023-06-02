from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from multiprocessing import Pool
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os, re
import torch
import nltk


class DataModule(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # text = torch.as_tensor(text, dtype=torch.float32)
        # label = torch.as_tensor(label, dtype=torch.float32)

        return text, label


class Cleanup():
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
        return out

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


class Preprocess():
    def __init__(self, stem: bool=False, lemma: bool=False):
        self.setup()
        self.stemmer = PorterStemmer() if stem else None
        self.lemmatizer = WordNetLemmatizer() if lemma else None
        self.stopwords = set(stopwords.words('english'))
    
    def __call__(self, text):
        return self.auto(text)
    
    def auto(self, text: str):
        tokens = self.tokenize(text)
        out = self.remove_stopwords(tokens)
        out = self.stemming(out)
        out = self.lemmatization(out)
        return ' '.join(out)
    
    def setup(self):
        nltk.download('stopwords', quiet=True)  # stopwords
        nltk.download('punkt', quiet=True)      # tokenize
        nltk.download('wordnet', quiet=True)    # stem & lemma
    
    def tokenize(self, text: str):
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: list):
        return [word for word in tokens if word not in self.stopwords]
    
    def stemming(self, tokens: list):
        if not self.stemmer:
            return tokens
        return [self.stemmer.stem(word) for word in tokens]
    
    def lemmatization(self, tokens: list):
        if not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(word) for word in tokens]


if __name__=='__main__':
    # Dataset
    dataset = pd.read_csv('datasets/IMDB.csv')
    data, label= dataset['review'], dataset['sentiment']

    # Define
    cleaner = Cleanup()
    preprocessser = Preprocess(stem=True, lemma=True)

    # Normal
    # tqdm.pandas()
    # data = data.progress_apply(cleaner)
    # data = data.progress_apply(preprocessser)

    # Multiprocessing (80% faster)
    num_cpu = os.cpu_count()
    with Pool(num_cpu) as pool:
        data = pool.map(cleaner, data)
        data = pool.map(preprocessser, data)

    print(data[:1])

