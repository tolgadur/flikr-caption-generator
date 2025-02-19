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


def train(epochs=10, batch=256, lr=0.001):
    # define the model and processor
    model = FlickrImageCaptioning(d_model=512, heads=8, n_layers=6).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # load the dataset
    ds = Flickr30kDataset()
    dataloader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.init(project="flickr30k-captioning")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        n_batches = 0

        for img, inp, tgt, mask in tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
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
            running_loss += loss.item()
            n_batches += 1

        # Compute epoch average loss
        epoch_loss = running_loss / n_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}")
        wandb.log({"loss": epoch_loss})

    scheduler.step()
    torch.save(model.state_dict(), "models/model.pth")
    wandb.save(f"models/model.pth")
