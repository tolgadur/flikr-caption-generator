import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

from model import FlickrImageCaptioning
from config import DEVICE
from dataset import Flickr30kDataset
from collate import collate_fn


def example_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    print(image_embeds.shape)
    print(text_embeds.shape)


def print_image():
    dataset = Flickr30kDataset()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )
    img, *_ = next(iter(dataloader))
    img = img.permute(0, 2, 3, 1).squeeze(0)
    plt.imshow(img)
    plt.show()


def load_model(model_path: str = "models/model.pth"):
    """Load the model from checkpoint and set to eval mode."""
    model = FlickrImageCaptioning().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
