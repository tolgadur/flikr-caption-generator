import torch
from transformers import CLIPProcessor
from PIL import Image
import requests
from config import DEVICE
from utils import load_model, display_image_with_caption
from dataset import Flickr30kDataset


def inference(
    image: Image.Image, model=None, temperature: float = 1.0, max_length: int = 77
):
    """Generate a caption for an image using the model."""
    if model is None:
        model = load_model()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    eos_token_id = processor.tokenizer.eos_token_id

    # Process image
    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs.pixel_values.to(DEVICE)
    # Start with no tokens - just use image embedding
    generated = None

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(pixel_values, generated)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1)  # shape: [1]

            if generated is None:
                generated = next_token.unsqueeze(-1)
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == eos_token_id:
                break

    caption = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption


def eval_sample():
    """Evaluate the model on a random training image."""
    dataset = Flickr30kDataset(split="train")
    image, gt_caption = dataset[1]

    caption = inference(image)
    display_image_with_caption(image, caption, gt_caption)


def eval_from_url(url: str):
    """Evaluate the model on an image from a URL."""
    image = Image.open(requests.get(url, stream=True).raw)
    caption = inference(image)
    display_image_with_caption(image, caption)
