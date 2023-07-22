import os, yaml
from argparse import ArgumentParser
from rich import traceback, print
from rich.prompt import Prompt
traceback.install()

from transformers import logging
logging.set_verbosity_error()

import torch
from modules.data import VnPreprocesser, CustomDataModule
from models import (
    RNN, LSTM, GRU, 
    BiRNN, BiLSTM, BiGRU, 
    BERT, GPT2
)



# General variable
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
MODEL = {
    "RNN": RNN, "LSTM": LSTM, "GRU": GRU,
    "BiRNN": BiRNN, "BiLSTM": BiLSTM, "BiGRU": BiGRU,
    "BERT": BERT, "GPT2": GPT2
}


def main(args, config):
    print("Starting...")

    prepare = VnPreprocesser(char_limit=7)

    config['data']['num_workers'] = NUM_WOKER
    dataset = CustomDataModule(**config['data'])

    model = MODEL[config['model']['model_name']](vocab_size=dataset.vocab_size, **config['model'])

    model.load(args.config['checkpoint'])
    model.eval().to(DEVICE)

    print("[bold]Started.[/]   ")

    while True:
        # Get input
        text = Prompt.ask("\n[bold]Enter prompt[/]") if not args.prompt else args.prompt

        # Prepare the text
        text = prepare(text)

        result = {"score": 0, "value": "Unidentified"}

        if text:
            # Encode
            text = dataset.encoder.encode([text]).to(DEVICE)

            # Make prediction
            with torch.inference_mode():
                result['score'] = model(text).item()

            # Format output
            if dataset.classes[round(result['score'])] == "POS":
                result['value'] = "[green]Positive[/]"
            else:
                result['value'] = "[red]Negative[/]" 

        # Print out the result
        print(f"[bold]Score:[/] {result['score']:.2f} -> [bold]{result['value']}[/]")

        # Exit if prompt argument is used
        exit() if args.prompt else None



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as file:
        args.config = yaml.full_load(file)['test']
        checkpoint_path = "/".join(args.config['checkpoint'].split("/")[:-2])
        checkpoint_config_path = checkpoint_path + '/hparams.yaml'
        with open(checkpoint_config_path, 'r', encoding='utf-8') as file:
            config = yaml.full_load(file)

    main(args, config)
