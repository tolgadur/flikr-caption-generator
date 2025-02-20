import torch

from trainer import train
from evals import eval_sample


torch.manual_seed(42)


def main():
    train(epochs=30)

    eval_sample()


if __name__ == "__main__":
    main()
