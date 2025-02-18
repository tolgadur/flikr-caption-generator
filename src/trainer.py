from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm

from dataset import Flickr30kDataset
from decoder import Decoder
from config import DEVICE
from collate import CLIPEmbedder, collate_fn


def train(epochs=10, batch=256, lr=0.001):
    # define the model and embedder
    model = Decoder(d_model=512, heads=8, n_layers=6).to(DEVICE)
    embedder = CLIPEmbedder()

    # load the dataset
    ds = Flickr30kDataset()
    dataloader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, embedder),
    )

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        n_batches = 0

        for _, inp_embeds, tgt_embeds in tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            inp_embeds = inp_embeds.to(DEVICE)
            tgt_embeds = tgt_embeds.to(DEVICE)

            optimizer.zero_grad()

            # forward pass
            outputs = model(inp_embeds)  # shape: [batch_size, seq_len, d_model]
            outputs = outputs[:, 1:, :]  # remove the first cls token
            loss = criterion(outputs, tgt_embeds)

            # backward pass
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            n_batches += 1

        # Compute epoch average loss
        epoch_loss = running_loss / n_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}")

        scheduler.step()
        torch.save(model.state_dict(), f"models/model_{epoch}.pth")
