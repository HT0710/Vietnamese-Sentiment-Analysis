from argparse import ArgumentParser
import os, yaml

import lightning_modules.data as data
import models
import torch

from rich.traceback import install
from rich import print
install()


DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = os.cpu_count() if DEVICE == 'cuda' else 0
CLASSES = ['Negative', 'Possitive']

config_path = "lightning_logs/2-10_256_Bi/hparams.yaml"
model_path = "lightning_logs/2-10_256_Bi/checkpoints/epoch=1-step=1500.ckpt"


def main(args):
    with open(config_path, 'r') as file:
        config = yaml.full_load(file)

    # Preprocessing
    preprocess = data.DataPreprocessing(**config["preprocess"])

    # Dataset
    config['data']['num_workers'] = NUM_WOKER
    dataset = data.IMDBDataModule(preprocessing=preprocess, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = models.RNNParallel(**config['model'])

    model.continue_from(model_path)
    model.to(DEVICE)
    model.eval()

    # Prepare data
    prepare = data.DataPreparation(lang='en', stopword=False, stem=True, lemma=True) 

    while True:
        text = input("> ") if not args.prompt else args.prompt

        text = prepare(text)

        if not text:
            continue

        text = preprocess.word2int(corpus=[text], vocab=preprocess.vocab)
        
        tensor = torch.as_tensor(text).to(DEVICE)

        with torch.inference_mode():
            result = model(tensor).item()
        
        print(f'{CLASSES[round(result)]} - ({result:.2f})', end="\n\n")

        exit() if args.prompt else None


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    args = parser.parse_args()
    main(args)
