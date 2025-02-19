from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm
from transformers import CLIPProcessor
import wandb
from dataset import Flickr30kDataset
from model import FlickrImageCaptioning
from config import DEVICE, VOCAB_SIZE
from collate import collate_fn


def validate(model, val_dataloader, criterion, epoch, epochs):
    """
    Run validation loop and return average validation loss.
    Args:
        model: The model to validate
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        epoch: Current epoch number
        epochs: Total number of epochs
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0
    val_batches = 0

    with torch.no_grad():
        for img, inp, tgt, mask in tqdm.tqdm(
            val_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
        ):
            img = img.to(DEVICE)
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            mask = mask.to(DEVICE)

            # forward pass
            outputs = model(img, inp, mask)
            outputs = outputs[:, 1:, :]

            outputs = outputs.reshape(-1, VOCAB_SIZE)
            tgt = tgt.reshape(-1)

            # compute loss
            loss = criterion(outputs, tgt)
            val_loss += loss.item()
            val_batches += 1

    return val_loss / val_batches


def train(epochs=10, batch=256, lr=0.002):
    # define the model and processor
    model = FlickrImageCaptioning(d_model=512, heads=8, n_layers=6).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # load the dataset
    train_ds = Flickr30kDataset(split="train")
    val_ds = Flickr30kDataset(split="val")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.init(project="flickr30k-captioning")
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0
        train_batches = 0

        for img, inp, tgt, mask in tqdm.tqdm(
            train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"
        ):
            img = img.to(DEVICE)
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            # forward pass
            outputs = model(img, inp, mask)
            outputs = outputs[:, 1:, :]  # remove the first token

            # reshape for cross entropy
            outputs = outputs.reshape(-1, VOCAB_SIZE)
            tgt = tgt.reshape(-1)

            # compute loss
            loss = criterion(outputs, tgt)

            # backward pass
            loss.backward()

            optimizer.step()

            # Update running loss
            train_loss += loss.item()
            train_batches += 1

        # Calculate losses
        train_epoch_loss = train_loss / train_batches
        val_epoch_loss = validate(model, val_dataloader, criterion, epoch, epochs)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {train_epoch_loss:.4f}")
        print(f"Validation Loss: {val_epoch_loss:.4f}")

        wandb.log({"train_loss": train_epoch_loss, "val_loss": val_epoch_loss})
        torch.save(model.state_dict(), f"models/model_{epoch}.pth")
        scheduler.step()

    torch.save(model.state_dict(), "models/model.pth")
    wandb.save("models/model.pth")
