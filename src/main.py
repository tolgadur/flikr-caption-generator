from trainer import train
import torch
from utils import print_image


torch.manual_seed(42)


def main():
    print_image()
    train()


if __name__ == "__main__":
    main()
