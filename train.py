import os
from argparse import ArgumentParser
import yaml

from models.GRU import GRU
from data import DataPreprocessing, IMDBDataModule

from lightning.pytorch import Trainer, seed_everything


def main(config):
    # Set seed
    seed_everything(seed=42, workers=True)

    # Preprocessing
    config['preprocess']['vocab'] = None  # change if use custom vocab
    preprocesser = DataPreprocessing(**config["preprocess"])

    # Dataset
    config['data']['num_workers'] = os.cpu_count()
    dataset = IMDBDataModule(preprocessing=preprocesser, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = GRU(**config['model'], save_hparams=config)

    # Trainer
    trainer = Trainer(max_epochs=config['trainer']['num_epochs'])

    trainer.fit(model, dataset)

    trainer.test(model, dataset, ckpt_path="best")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)

    if args.epoch is not None:
        config['trainer']['num_epochs'] = args.epoch
    if args.batch is not None:
        config['data']['batch_size'] = args.batch
    if args.learning_rate is not None:
        config['trainer']['learning_rate'] = args.learning_rate

    main(config)
