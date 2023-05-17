import os
from argparse import ArgumentParser

from models import RNN, LSTM, GRU
from model import LitModel
from data import DataPreprocessing, IMDBDataModule

import torch
from lightning.pytorch import Trainer, seed_everything


def main(args):
    # Set seed
    seed_everything(seed=42, workers=True)

    # Hyperparameter
    hparams = {
        "preprocesser": {
            "vocab": None,
            "seq_length": 400,
            "min_freq": 2,
            "max_freq": 0.5,
        },
        "dataset": {
            "train_val_test_split": (0.75, 0.1, 0.15),
            "batch_size": args.batch,
            "num_workers": os.cpu_count(),
            "pin_memory": True,
        },
        "model": {
            "output_size": 1,
            "hidden_size": 512,
            "embedding_size": 300,
            "n_layers": 2,
            "dropout": 0.25,
        }
    }

    # Preprocessing
    preprocesser = DataPreprocessing(**hparams["preprocesser"])

    # Dataset
    dataset = IMDBDataModule(
        data_path='datasets/IMDB.csv',
        download=False,
        preprocessing=preprocesser,
        **hparams["dataset"]
    )
    hparams["model"]["vocab_size"] = dataset.vocab_size

    # Model
    net = GRU(**hparams["model"])

    hparams["model"]["name"] = net._get_name()

    compile = torch.compile(net)

    model = LitModel(model=compile, lr=args.learning_rate, save_hparams=hparams)

    # Train
    trainer = Trainer(max_epochs=args.epoch)

    trainer.fit(model, dataset)

    trainer.test(model, dataset, ckpt_path="best")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default = 10)
    parser.add_argument("-b", "--batch", type=int, default = 32)
    parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-3)
    args = parser.parse_args()

    main(args)
