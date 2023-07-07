import os, re, requests
from typing import Union, List

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import py_vncorenlp as vncore
import underthesea as uts
from pyvi import ViUtils



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
        original = os.getcwd()
        vncore_path = f"{original}/models/vncorenlp"
        if not os.path.exists(vncore_path):
            os.mkdir(vncore_path)
            vncore.download_model(save_dir=vncore_path)
        tokenizer = vncore.VnCoreNLP(annotators=["wseg"], save_dir=vncore_path)
        os.chdir(original)
        return tokenizer

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
