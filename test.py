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

config_path = "lightning_logs/version_0/hparams.yaml"
model_path = "lightning_logs/version_0/checkpoints/epoch=2-step=1101.ckpt"


def main(args):
    print("Starting..|", end="\r")
    with open(config_path, 'r') as file:
        config = yaml.full_load(file)

    # Preprocessing
    preprocess = data.DataPreprocessing(**config["preprocess"])

    print("Starting../", end="\r")
    # Dataset
    config['data']['num_workers'] = NUM_WOKER
    dataset = data.CustomDataModule(
        data_path="datasets/dataset_t1s1a1.csv",
        preprocessing=preprocess,
        **config['data']
    )
    print("Starting..-", end="\r")

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = models.GRU(**config['model'])
    print("Starting../", end="\r")

    model.continue_from(model_path)
    model.to(DEVICE)
    model.eval()
    print("Starting..-", end="\r")

    # Prepare data
    prepare = data.VnPreparation(char_limit=7)
    print("Started.   ")

    while True:
        if not args.prompt:
            print("\n[bold]Enter prompt:[/]", end=" ")
            text = input()
        else:
            text = args.prompt

        text = prepare(text)

        if not text:
            print("[bold]Score:[/] 0 -> [bold]Unidentified[/]")
            continue

        text = preprocess.word2int(corpus=[text], vocab=preprocess.vocab)

        tensor = torch.as_tensor(text).to(DEVICE)

        with torch.inference_mode():
            output = model(tensor).item()

        if 0.4 < output < 0.6:
            print(f"[bold]Score:[/] {output:.2f} -> [bold][yellow]Neutral[/][/]")
            continue

        result = CLASSES[round(output)]
        check = lambda x: f"[red]{x}[/]" if x == 'Negative' else f"[green]{x}[/]" 

        print(f"[bold]Score:[/] {output:.2f} -> [bold]{check(result)}[/]")

        exit() if args.prompt else None


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    args = parser.parse_args()
    main(args)
