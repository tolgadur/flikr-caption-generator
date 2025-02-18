from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tqdm

from dataset import Flickr30kDataset
from decoder import Decoder
from config import DEVICE


def train(epochs=10, batch_size=256, lr=0.001):
    # define the model
    model = Decoder(d_model=512, heads=8, n_layers=6).to(DEVICE)

    # load the dataset
    ds = Flickr30kDataset()
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        for _, image_embeds, input_seq, target_seq in tqdm.tqdm(dataloader):
            image_embeds = image_embeds.to(DEVICE)
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)

            optimizer.zero_grad()

            # forward pass
            outputs = model(image_embeds, input_seq)
            loss = criterion(outputs, target_seq)

            # backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
            epoch_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")
        torch.save(model.state_dict(), f"models/model_{epoch}.pth")
