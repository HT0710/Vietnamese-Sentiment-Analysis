import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np



def build_vocabulary(data: list):
    words = ' '.join(data).split(' ')                       # Join all the text data and split it into individual words
    counter = Counter(words)                                # Count the occurrences of each word
    vocab = sorted(counter, key=counter.get, reverse=True)  # Sort the words in descending order of frequency
    int2word = dict(enumerate(vocab, 1))                    # Create a mapping from index to word (int to word)
    int2word[0] = '<PAD>'                                   # Add a special token for padding at index 0
    word2int = {word: id for id, word in int2word.items()}  # Create a mapping from word to index (word to int)
    return word2int


def encode_words(data: list, vocab: dict):
    encoded = []
    for seq in tqdm(data, 'Encode'):
        seq_idx = []
        for word in seq.split():
            seq_idx.append(vocab[word])
        encoded.append(seq_idx)
    return encoded


def padding(data, pad_id, seq_length=128):
    features = np.full((len(data), seq_length), pad_id, dtype=int)
    for i, row in enumerate(data):
        features[i, :len(row)] = np.array(row)[:seq_length]
    return features


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = 'cpu'
):
    """Train function"""

    # Train mode
    model.train()

    # Define
    train_loss, train_acc = 0, 0
    data_size = len(dataloader)

    # Train loop
    for step, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.to(device)

        outputs = model(X).squeeze()
        loss = criterion(outputs, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, y, task='binary')

        print(f'  |- Train step: {step}/{data_size} | Loss: {train_loss/step:.4f}  Acc: {train_acc/step:.4f}     ', end='\r')

    train_loss = train_loss / data_size
    train_acc = train_acc / data_size

    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device = 'cpu'
):
    """Test function"""

    # Evaluate mode
    model.eval()

    # Define
    test_loss, test_acc = 0, 0
    data_size = len(dataloader)

    # Evaluate loop
    with torch.inference_mode():
        for step, (X, y) in enumerate(dataloader, 1):
            X, y = X.to(device), y.to(device)

            outputs = model(X).squeeze()
            loss = criterion(outputs, y.float())

            test_loss += loss.item()
            test_acc += accuracy(outputs, y, task='binary')

            print(f'  |- Test step: {step}/{data_size} | Loss: {test_loss/step:.4f}  Acc: {test_acc/step:.4f}     ', end='\r')

    test_loss = test_loss / data_size
    test_acc = test_acc / data_size

    return test_loss, test_acc


def save_model(model: nn.Module, target_dir: str, model_name: str):
    """Save model"""
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(), f=model_save_path)
