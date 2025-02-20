import torch
from transformers import CLIPProcessor
from PIL import Image
import requests
from config import DEVICE, MODEL
from utils import display_image_with_caption
from dataset import Flickr30kDataset
import random


def inference(
    image: Image.Image,
    model=MODEL,
    max_length: int = 77,
    temperature: float = 1.0,
):
    """Generate a caption for an image using the model.

    Args:
        image: PIL Image to generate caption for
        model: Model to use for inference
        max_length: Maximum length of generated caption
        temperature: higher = more diverse, lower = more conservative
    """
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
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # Sample from the distribution instead of taking argmax
            next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(-1)

            if generated is None:
                generated = next_token.unsqueeze(-1)
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == eos_token_id:
                break

    caption = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption


def sample_five_inference(
    image: Image.Image, model=MODEL, max_length: int = 77
) -> list[str]:
    """Generate five different captions for an image using different temperature values.

    Args:
        image: PIL Image to generate captions for
        model: Model to use for inference
        max_length: Maximum length of generated captions

    Returns:
        List of 5 generated captions
    """

    # Always include temperature=1.0, and generate 4 random temperatures between 0 and 1.5
    temperatures = [1.0]
    for _ in range(4):
        temperatures.append(random.uniform(0, 1.5))

    captions = []
    for temp in temperatures:
        caption = inference(image, model=model, max_length=max_length, temperature=temp)
        captions.append(caption)

        print(f"Temperature: {temp}")
        print(f"Caption: {caption}")
        print("-" * 100)

    return captions


def eval_sample():
    """Evaluate the model on a random training image."""
    dataset = Flickr30kDataset(split="test")

    for i in range(10):
        image, gt_caption = dataset[i]

        captions = sample_five_inference(image)
        print(captions)
        display_image_with_caption(image, captions, gt_caption)


def eval_from_url(url: str):
    """Evaluate the model on an image from a URL."""
    image = Image.open(requests.get(url, stream=True).raw)
    caption = inference(image)
    display_image_with_caption(image, caption)
