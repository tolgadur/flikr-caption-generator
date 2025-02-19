import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Flickr30kDataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
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
    images_plt, *_ = next(iter(dataloader))
    image = images_plt.squeeze(0)  # remove batch
    plt.imshow(image)
    plt.show()
