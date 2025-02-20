import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

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


def print_example_image():
    """Display a random image from the dataset using the dataloader."""
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


def display_image_with_caption(
    image: Image.Image, caption: str, ground_truth: str = None
):
    """Display an image with its caption(s)."""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")

    title = f"Generated: {caption}"
    if ground_truth:
        title += f"\nGround truth: {ground_truth}"

    plt.title(title, wrap=True)
    plt.show()
