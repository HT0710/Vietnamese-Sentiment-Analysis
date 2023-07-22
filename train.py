import os, yaml
from argparse import ArgumentParser
from rich import traceback
traceback.install()

from transformers import logging
logging.set_verbosity_error()

import torch
from lightning.pytorch import Trainer, seed_everything
from modules.callback import callbacks_list
from modules.data import CustomDataModule
from models import (
    RNN, LSTM, GRU, 
    BiRNN, BiLSTM,  BiGRU,
    BERT, GPT2
)



# Set seed
seed_everything(seed=42, workers=True)
# Handle forked process (set to `false` if process is stuck)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Set number of worker (CPU will be used | Default: 80%)
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
MODEL = {
    "RNN": RNN, "LSTM": LSTM, "GRU": GRU,
    "BiRNN": BiRNN, "BiLSTM": BiLSTM, "BiGRU": BiGRU,
    "BERT": BERT, "GPT2": GPT2
}

def main(config):
    config['data']['num_workers'] = NUM_WOKER

    # Dataset
    dataset = CustomDataModule(**config['data'])

    # Model
    model = MODEL[config['model']['model_name']](vocab_size=dataset.vocab_size, **config['model'])

    # Load checkpoint if configured
    model.load(config['trainer']['checkpoint'])

    # Save hyperparameters
    model.save_hparams(config)

    # Define trainer
    trainer = Trainer(
        max_epochs=config['trainer']['num_epochs'],
        callbacks=callbacks_list(config['callback'])
    )

    # Training
    trainer.fit(model, dataset)

    # Testing
    trainer.test(model, dataset)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-e", "--epoch", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)['train']

    # Overwrite config if arguments is not None
    if args.model is not None:
        config['model']['model_name'] = args.model
    if args.epoch is not None:
        config['trainer']['num_epochs'] = args.epoch
    if args.batch is not None:
        config['data']['batch_size'] = args.batch
    if args.learning_rate is not None:
        config['trainer']['learning_rate'] = args.learning_rate
    if args.checkpoint is not None:
        config['trainer']['checkpoint'] = args.checkpoint

    main(config)
