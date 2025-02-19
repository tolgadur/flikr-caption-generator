import torch
from datasets import load_dataset


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        ds = load_dataset("nlphuji/flickr30k", split="test").select(range(100))
        self.ds = ds.filter(lambda x: x["split"] == split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        # For now just taking first caption, but could return all captions
        text = item["caption"][0]
        return image, text
