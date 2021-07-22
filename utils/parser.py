import argparse
import numpy as np
import torch

from datetime import datetime
from pathlib import Path


def set_seed(seed: int):
    if seed != None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Parser(object):
    @staticmethod
    def train() -> argparse.Namespace:

        parser = argparse.ArgumentParser()

        general = parser.add_argument_group("general")
        general.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
        general.add_argument("--name", type=str, default=datetime.now().strftime("%d:%m_%H:%M:%S"),
                             help="name of directory for product")

        dataset = parser.add_argument_group("dataset")
        dataset.add_argument("--train-file", type=Path, required=True, help="path to train file")
        dataset.add_argument("--max-seq-len", type=int, default=75, help="maximal sequence length")

        training = parser.add_argument_group("training")
        training.add_argument(
            "--finetune",
            action="store_true",
            default=False,
            help="set to finetune classifier rather than train entire model",
        )
        training.add_argument("--num-epochs", type=int, default=25, help="number of epochs to train")
        training.add_argument("--batch-size", type=int, default=32, help="batch size")

        optimizer = parser.add_argument_group("optimizer")
        optimizer.add_argument("--learning-rate", type=float, default=3e-5, help="learning rate")
        optimizer.add_argument("--optimizer-eps", type=float, default=1e-8, help="optimizer tolerance")
        optimizer.add_argument("--weight-decay-rate", type=float, default=0.01, help="optimizer weight decay rate")
        optimizer.add_argument("--max-grad-norm", type=float, default=1.0, help="maximal gradients norm")

        scheduler = parser.add_argument_group("scheduler")
        scheduler.add_argument("--num-warmup-steps", type=int, default=0, help="scheduler warmup steps")

        opts = parser.parse_args()
        set_seed(opts.seed)
        opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return opts
