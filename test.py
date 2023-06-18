from argparse import ArgumentParser
import os, yaml

import lightning_modules.data as data
import models
import torch

from rich import print
from rich.prompt import Prompt
from rich.traceback import install
install()


DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
NUM_WOKER = os.cpu_count() if DEVICE == 'cuda' else 0

result_format = {
    "NEG": "[red]Negative[/]",
    "NEU": "[yellow]Neutral[/]",
    "POS": "[green]Positive[/]",
}


def main(args):
    print("Starting...", end="\r")

    # Load config
    model_path = args.config['test']['model_path']
    config_path = args.config['test']['config_path']
    with open(config_path, 'r') as file:
        config = yaml.full_load(file)

    # Preprocessing
    preprocess = data.DataPreprocessing(**config["preprocess"])

    # Dataset
    config['data']['num_workers'] = NUM_WOKER
    dataset = data.CustomDataModule(preprocessing=preprocess, **config['data'])

    # Model
    config['model']['vocab_size'] = dataset.vocab_size
    model = models.BiGRU(**config['model'])

    model.load(model_path)
    model.to(DEVICE)
    model.eval()

    # Prepare data
    prepare = data.VnPreparation(char_limit=7)
    print("[bold]Started.[/]   ")

    while True:
        text = args.prompt if args.prompt else Prompt.ask("\n[bold]Enter prompt[/]")

        text = prepare(text)

        result = {"score": 0, "value": None}

        if text:
            text = preprocess.word2int(corpus=[text], vocab=preprocess.vocab)

            tensor = torch.as_tensor(text).to(DEVICE)

            with torch.inference_mode():
                output = model(tensor).item()

            result = {
                "score": output,
                "value": "NEU" if (0.4 < output < 0.6) else dataset.classes[round(output)]
            }

        print(f"[bold]Score:[/] {round(result['score'], 2)} -> [bold]{result_format.get(result['value'], 'Unidentified')}[/]")

        exit() if args.prompt else None


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        args.config = yaml.full_load(file)

    main(args)
