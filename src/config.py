import torch
from transformers import CLIPProcessor


def get_vocab():
    pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    word2index = pr.tokenizer.get_vocab()
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

TOKEN2INDEX, INDEX2TOKEN = get_vocab()

print(f"Using device: {DEVICE}")
