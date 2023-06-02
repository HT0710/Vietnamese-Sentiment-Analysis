from utils import build_vocabulary, word2int
from data import Cleanup, Preprocess
from models.GRU import GRU
import pandas as pd
import torch
import os


dataset = pd.read_csv('datasets/IMDB_processed.csv')
data, label = dataset['text'].tolist(), dataset['label'].tolist()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = os.cpu_count() if DEVICE == 'cuda' else 0

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYER = 2
DROPOUT = 0.25

# Build vocabulary
vocab = build_vocabulary(data)

model = GRU(
    vocab_size=len(vocab),
    output_size=1,
    hidden_size=HIDDEN_SIZE,
    embedding_size=EMBEDDING_SIZE,
    n_layers=NUM_LAYER,
    dropout=DROPOUT
).to(DEVICE)

model.load_state_dict(torch.load("checkpoints/GRU/E5_L0.2950_A0.88.pth", map_location=DEVICE))

cleaner = Cleanup()
preprocesser = Preprocess(True, True)
CLASSES = ['Negative', 'Possitive']

while True:
    text = input("\n> ")
    
    text = cleaner(text)
    text = preprocesser(text)
    try:
        text = word2int([text], vocab)
    except:
        print("Out of vocab")
        continue

    tensor = torch.as_tensor(text).to(DEVICE)
    
    with torch.inference_mode():
        result = model(tensor).item()

    print(f'{result:.2f} - ({CLASSES[round(result)]})     ')
