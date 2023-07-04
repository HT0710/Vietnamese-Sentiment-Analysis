import os, yaml
from argparse import ArgumentParser
from rich import traceback, print
from rich.prompt import Prompt
traceback.install()

import torch
from modules.data import VnPreprocesser, CustomDataModule
from models import (
    RNN, LSTM, GRU, 
    BiRNN, BiGRU, BiLSTM, 
    BERT, GPT2
)



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
    with open(args.config['test']['config_path'], 'r', encoding='utf-8') as file:
        config = yaml.full_load(file)

    prepare = VnPreprocesser(char_limit=7)

    config['data']['num_workers'] = NUM_WOKER
    dataset = CustomDataModule(**config['data'])

    model = GPT2(**config['model'])

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
            text = dataset.encoder(text).to(DEVICE)

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
