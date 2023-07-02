from argparse import ArgumentParser
import os, yaml

import lightning_modules.data as data
import models
import torch

from rich import print
from rich.prompt import Prompt
from rich.traceback import install
install()


# General variable
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
FORMAT = {
    "NEG": "[red]Negative[/]",
    "NEU": "[yellow]Neutral[/]",
    "POS": "[green]Positive[/]",
}


def main(args):
    print("Starting...", end="\r")

    # Load config
    with open(args.config['test']['config_path'], 'r') as file:
        config = yaml.full_load(file)

    prepare = data.VnPreprocesser(char_limit=7)

    encoder = data.CustomEncoder.load("vinai/phobert-base-v2")

    config['data']['num_workers'] = NUM_WOKER
    dataset = data.CustomDataModule(encoder=encoder, **config['data'])

    config['model']['vocab_size'] = dataset.vocab_size
    model = models.BERT(**config['model'])

    model.load(args.config['test']['model_path'])
    model.to(DEVICE)
    model.eval()

    print("[bold]Started.[/]   ")

    while True:
        # Get input
        text = args.prompt if args.prompt else Prompt.ask("\n[bold]Enter prompt[/]")

        # Prepare the text
        text = prepare(text)

        result = {"score": 0, "value": None}

        if text:
            # Make prediction
            text = encoder(text).to(DEVICE)

            with torch.inference_mode():
                output = model(text).item()

            result = {
                "score": output,
                "value": "NEU" if (0.4 < output < 0.6) else dataset.classes[round(output)]
            }

        # Print out the result
        print(f"[bold]Score:[/] {round(result['score'], 2)} -> [bold]{FORMAT.get(result['value'], 'Unidentified')}[/]")

        # Exit if prompt argument is used
        exit() if args.prompt else None


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        args.config = yaml.full_load(file)

    main(args)
