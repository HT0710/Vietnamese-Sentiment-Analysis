from multiprocessing import Pool
from lightning_modules import data
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

prepare = data.VnPreparation(char_limit=7, **CONFIG)

with Pool(NUM_WOKERS) as pool:
    text = pool.map(prepare, dataset['text'])

dataset['text'] = text

dataset = dataset[dataset != ''].dropna()
dataset = dataset[dataset['text'].apply(lambda x: len(str(x).split(" ")) <= 200)]
dataset = dataset.drop_duplicates(keep='first')

dataset.to_csv(SAVE_PATH, index=False)
