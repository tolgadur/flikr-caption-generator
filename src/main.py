from trainer import train
import torch


torch.manual_seed(42)


def main():
    train(epochs=20)


if __name__ == "__main__":
    main()
