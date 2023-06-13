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
    dataset = data.VietDataModule(preprocessing=preprocesser, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = models.GRU(**config['model'])
    model.save_hparams(config)

    # Continue
    if config['trainer']['weight']:
        model.load(config['trainer']['weight'])

    # Trainer
    trainer = Trainer(
        logger=True,
        max_epochs=config['trainer']['num_epochs'],
        callbacks=callbacks_list(config['callback']),
        enable_checkpointing=config['callback']['checkpoint']['enable'],
    )

    trainer.fit(model, dataset)

    trainer.test(model, dataset)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=None)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-w", "--weight", type=str, default=None)
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.full_load(file)

    if args.epoch:
        config['trainer']['num_epochs'] = args.epoch
    if args.batch:
        config['data']['batch_size'] = args.batch
    if args.learning_rate:
        config['trainer']['learning_rate'] = args.learning_rate
    if args.weight:
        config['trainer']['weight'] = args.weight

    main(config)
