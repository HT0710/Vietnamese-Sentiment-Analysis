from multiprocessing import Pool
from modules.data import VnPreprocesser
import pandas as pd
import os


CONFIG = {
    "tokenize": True,
    "stopwords": True,
    "accents": True
}
SAVE_PATH = "datasets/dataset_t{}s{}a{}.csv".format(
    int(CONFIG['tokenize']), int(CONFIG['stopwords']), int(CONFIG['accents'])
)
NUM_WOKERS = int(os.cpu_count()*0.8)


dataset_raw = pd.read_csv('datasets/dataset_raw.csv')
dataset = dataset_raw.copy()

prepare = VnPreprocesser(char_limit=7, **CONFIG)

with Pool(NUM_WOKERS) as pool:
    text = pool.map(prepare, dataset['text'])

dataset['text'] = text

dataset = dataset[dataset != ''].dropna()
dataset = dataset.drop_duplicates(keep='first')

dataset.to_csv(SAVE_PATH, index=False)
