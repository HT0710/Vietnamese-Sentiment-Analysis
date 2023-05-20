import os
from argparse import ArgumentParser
import yaml

from models import RNN, LSTM, GRU
from model import LitModel
from data import DataPreprocessing, IMDBDataModule

import torch
from lightning.pytorch import Trainer, seed_everything


def main(args):
    # Set seed
    seed_everything(seed=42, workers=True)

    # Preprocessing
    config['preprocess']['vocab'] = None  # overwrite if use custom vocab
    preprocesser = DataPreprocessing(**config["preprocess"])

    # Dataset
    config['data'].update({
        'batch_size': args.batch,
        'num_workers': os.cpu_count(),
    })
    dataset = IMDBDataModule(preprocessing=preprocesser, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    
    net = GRU(**config['model'])
    compile = torch.compile(net)

    config['model']['name'] = net._get_name()

    # Lightning
    lr = config['trainer']['learning_rate'] = args.learning_rate
    model = LitModel(model=compile, lr=lr, save_hparams=config)

    # Train
    epochs = config['trainer']['epochs'] = args.epoch
    trainer = Trainer(max_epochs=epochs)

    trainer.fit(model, dataset)

    trainer.test(model, dataset, ckpt_path="best")


if __name__=="__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default = config['trainer']['epochs'])
    parser.add_argument("-b", "--batch", type=int, default = config['data']['batch_size'])
    parser.add_argument("-lr", "--learning_rate", type=float, default = config['trainer']['learning_rate'])
    args = parser.parse_args()

    main(args)
