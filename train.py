from argparse import ArgumentParser
import os, yaml

from lightning.pytorch import Trainer, seed_everything
from lightning_modules import callbacks_list
import lightning_modules.data as data
import models
import torch

from rich.traceback import install
install()


# Set seed
seed_everything(seed=42, workers=True)


def main(config):
    # Preprocessing
    preprocesser = data.DataPreprocessing(**config["preprocess"])

    # Dataset
    config['data']['num_workers'] = os.cpu_count() if torch.cuda.is_available() else 0
    dataset = data.CustomDataModule(preprocessing=preprocesser, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = models.BiGRU(**config['model'])
    model.save_hparams(config)
    model.load(config['trainer']['checkpoint'])

    # Trainer
    trainer = Trainer(
        max_epochs=config['trainer']['num_epochs'],
        callbacks=callbacks_list(config['callback'])
    )

    # Training and testing
    trainer.fit(model, dataset)

    trainer.test(model, dataset)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)
        config = config['train']

    # Overwrite config if arguments is not None
    if args.epoch is not None:
        config['trainer']['num_epochs'] = args.epoch
    if args.batch is not None:
        config['data']['batch_size'] = args.batch
    if args.learning_rate is not None:
        config['trainer']['learning_rate'] = args.learning_rate
    if args.checkpoint is not None:
        config['trainer']['checkpoint'] = args.checkpoint

    main(config)
