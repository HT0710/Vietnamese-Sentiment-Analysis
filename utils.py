from torchtext.functional import truncate, pad_sequence
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
from collections import Counter
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
import os, torch


def build_vocabulary(data: list[list]):
    overprint("Building vocabulary...", replaced=True)
    words = ' '.join(data).split(' ')                       # Join all the text data and split it into individual words
    counter = Counter(words)                                # Count the occurrences of each word
    vocab = sorted(counter, key=counter.get, reverse=True)  # Sort the words in descending order of frequency
    int2word = dict(enumerate(vocab, 0))                    # Create a mapping from index to word (int to word)
    vocab = {word: id for id, word in int2word.items()}     # Create a mapping from word to index (word to int)
    return vocab


def word2int(data: list[list], vocab: dict):
    overprint("Converting word2int...", replaced=True)
    return [[vocab[word] for word in seq.split()] for seq in data]


def truncate_sequences(data: list[list], seq_length=128):
    overprint("Truncating...", replaced=True)
    return truncate(data, seq_length)


def pad_sequences(data: list[list]):
    overprint("Padding...", replaced=True)
    to_tensor = [torch.as_tensor(seq) for seq in data]
    return pad_sequence(to_tensor, batch_first=True)


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
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(), f=model_save_path)


def overprint(string: str, replaced: bool=True):
    start = f"{' '*os.get_terminal_size()[0]}\r{string}"
    end = '\r' if replaced else '\n'
    return print(start, end=end)
