from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tqdm

from dataset import Flickr30kDataset
from decoder import Decoder
from config import DEVICE


def collate_fn(batch):
    # Separate the batch components
    images, img_embeds, inp_embeds, tgt_embeds = zip(*batch)

    # Pad sequences to max length (automatically handles different lengths)
    inp_embeds = pad_sequence([x.squeeze(0) for x in inp_embeds], batch_first=True)
    tgt_embeds = pad_sequence([x.squeeze(0) for x in tgt_embeds], batch_first=True)

    return images, img_embeds, inp_embeds, tgt_embeds


def train(epochs=10, batch=256, lr=0.001):
    # define the model
    model = Decoder(d_model=512, heads=8, n_layers=6).to(DEVICE)

    # load the dataset
    ds = Flickr30kDataset()
    dataloader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_fn)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        for _, img_embeds, inp_embeds, tgt_embeds in tqdm.tqdm(dataloader):
            img_embeds = img_embeds.to(DEVICE)
            inp_embeds = inp_embeds.to(DEVICE)
            tgt_embeds = tgt_embeds.to(DEVICE)

            optimizer.zero_grad()

            # forward pass
            inp = torch.cat((img_embeds, inp_embeds), dim=1)
            outputs = model(inp)  # shape: [batch_size, seq_len, d_model]
            outputs = outputs[:, 1:, :]  # remove the first cls token
            loss = criterion(outputs, tgt_embeds)

            # backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
            epoch_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")
        torch.save(model.state_dict(), f"models/model_{epoch}.pth")
