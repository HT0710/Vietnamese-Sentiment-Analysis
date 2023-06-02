from utils import train_step, test_step, save_model, build_vocabulary, word2int, truncate_sequences, pad_sequences
from torch.utils.data import DataLoader, random_split
from models import RNN, LSTM, GRU
from data import DataModule
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os, math
import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = os.cpu_count() if DEVICE == 'cuda' else 0

# Impact performance
TRAIN_TEST_SIZE = (0.8, 0.2)
SEQ_LENGTH = 400
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYER = 2
DROPOUT = 0.25
BATCH_SIZE = 200
LEARNING_RATE = 0.001
EPOCHS = 20

# Load dataset
dataset = pd.read_csv('datasets/IMDB_processed.csv')
data, label = dataset['text'].tolist(), dataset['label'].tolist()

# Build vocabulary
vocab = build_vocabulary(data)

# Encode words
encoded = word2int(data, vocab)

# Truncate sequences
truncated = truncate_sequences(encoded, seq_length=SEQ_LENGTH)

# Padding sequences
padded = pad_sequences(truncated)

# Create data module
dataset = DataModule(padded, label)

# Train, test split
train_data, test_data = random_split(dataset, TRAIN_TEST_SIZE)

# Create dataloader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WOKER)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WOKER)

# Model
model = GRU(
    vocab_size=len(vocab),
    output_size=1,
    hidden_size=HIDDEN_SIZE,
    embedding_size=EMBEDDING_SIZE,
    n_layers=NUM_LAYER,
    dropout=DROPOUT
).to(DEVICE)

# Loss, optimizer and learning rate scheduler
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

# Others
best_loss = {
    'name': None,
    'value': math.inf
}
earlystopping = {
    'patient': 3,
    'count': 0
}

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1} | Lr: {optimizer.param_groups[0]['lr']}")

    train_loss, train_acc = train_step(
        model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=DEVICE
    )

    test_loss, test_acc = test_step(
        model=model, dataloader=test_loader, criterion=criterion, device=DEVICE
    )

    # scheduler.step(test_loss)

    print(f"  |- Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | Test_loss: {test_loss:.4f}  Test_acc: {test_acc:.4f}")

    if test_loss < best_loss['value']:
        save_dir = f"checkpoints/{model._get_name()}"
        if best_loss['name'] is not None:
            os.remove(f"{save_dir}/{best_loss['name']}")
        best_loss['value'] = test_loss
        best_loss['name'] = f"E{epoch+1}_L{best_loss['value']:.4f}_A{test_acc:.2f}.pth"
        save_model(model=model, target_dir=save_dir, model_name=best_loss['name'])
        earlystopping['count'] = 0
    else:
        earlystopping['count'] += 1
        if earlystopping['count'] == earlystopping['patient']:
            exit("Exit by Early Stopping.")
