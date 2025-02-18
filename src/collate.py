import torch
from transformers import CLIPProcessor
from typing import List, Tuple
from PIL.Image import Image


class CLIPEmbedder:
    def __init__(self):
        self.pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, imgs: List[Image], txts: List[str]) -> Tuple[torch.Tensor, ...]:
        # Process the batch with padding
        enc = self.pr(images=imgs, text=txts, return_tensors="pt", padding=True)

        # Get tokenized inputs and pixel values
        tkn = enc["input_ids"]  # shape [batch_size, seq_len + 2] for bos and eos
        img = enc["pixel_values"]  # shape [batch_size, 3, 224, 224]
        mask = enc["attention_mask"][:, 1:]  # shape [batch_size, seq_len + 1]

        # Create input and target sequences for training
        inp = tkn[:, 1:]  # shape [batch_size, seq_len + 1]
        tgt = tkn[:, :-1]  # shape [batch_size, seq_len + 1]

        return img, inp, tgt, mask


def collate_fn(
    batch: List[Tuple[Image, str]], embedder: CLIPEmbedder
) -> Tuple[torch.Tensor, ...]:
    """
    Collate function to be used with DataLoader.
    Args:
        batch: List of tuples (image, text)
        embedder: Instance of CLIPEmbedder
    Returns:
        Tuple of tensors (pixel_values, input_ids, target_ids, mask)
    """
    images, texts = zip(*batch)
    return embedder(list(images), list(texts))
