import torch
from transformers import CLIPProcessor
from typing import List, Tuple
from PIL.Image import Image


def collate_fn(
    batch: List[Tuple[Image, str]], processor: CLIPProcessor
) -> Tuple[torch.Tensor, ...]:
    """
    Collate function to be used with DataLoader.
    Args:
        batch: List of tuples (image, text)
        processor: CLIP processor instance
    Returns:
        Tuple of tensors (pixel_values, input_ids, target_ids, mask)
    """
    images, texts = zip(*batch)

    # Process the batch with padding
    enc = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Get tokenized inputs and pixel values
    tkn = enc["input_ids"]  # shape [batch_size, seq_len + 2] for bos and eos
    img = enc["pixel_values"]  # shape [batch_size, 3, 224, 224]
    mask = enc["attention_mask"][:, :-1]  # shape [batch_size, seq_len + 1]

    # Create input and target sequences for training
    inp = tkn[:, 1:-1]  # shape [batch_size, seq_len]
    tgt = tkn[:, 1:]  # shape [batch_size, seq_len + 1]

    return img, inp, tgt, mask
