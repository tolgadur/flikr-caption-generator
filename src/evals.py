import torch
from PIL import Image
import requests
from config import DEVICE, MODEL, CLIP_MODEL, CLIP_PROCESSOR
from utils import display_image_with_caption
from dataset import Flickr30kDataset
from schemas import CaptionDetails, CaptionResponse


def inference(
    image: Image.Image,
    model=MODEL,
    max_length: int = 77,
    temperature: float = 1.0,
    use_argmax: bool = False,
):
    """Generate a caption for an image using the model.

    Args:
        image: PIL Image to generate caption for
        model: Model to use for inference
        max_length: Maximum length of generated caption
        temperature: higher = more diverse, lower = more conservative
        use_argmax: If True, use argmax instead of sampling
    """
    eos_token_id = CLIP_PROCESSOR.tokenizer.eos_token_id

    # Process image
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")

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

            # Choose next token based on strategy
            if use_argmax:
                next_token = torch.argmax(next_token_probs, dim=-1)
            else:
                next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(
                    -1
                )

            if generated is None:
                generated = next_token.unsqueeze(-1)
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == eos_token_id:
                break

    caption = CLIP_PROCESSOR.tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption


def sample_five_inference(
    image: Image.Image, model=MODEL, max_length: int = 77
) -> list[CaptionDetails]:
    """Generate different captions using both argmax and sampling strategies."""
    configs = [
        (1.0, True),  # argmax, standard temp
        (1.0, False),  # sampling, standard temp
        (0.1, False),  # sampling, conservative
        (0.01, False),  # sampling, very conservative
        (0.5, False),  # sampling, moderate
        (0.7, False),  # sampling, slightly conservative
        (1.5, False),  # sampling, creative
    ]

    caption_details = []

    for temp, use_argmax in configs:
        caption = inference(
            image,
            model=model,
            max_length=max_length,
            temperature=temp,
            use_argmax=use_argmax,
        )

        details = CaptionDetails(
            caption=caption,
            temperature=temp,
            is_argmax=use_argmax,
        )
        caption_details.append(details)

        print(f"Temperature: {temp}, {'Argmax' if use_argmax else 'Sampling'}")
        print(f"Caption: {caption}")
        print("-" * 100)

    return caption_details


def get_best_caption(
    image: Image.Image, model=MODEL, max_length: int = 77
) -> CaptionResponse:
    """Generate captions and return the one most aligned with the image.

    Uses CLIP to compare the generated captions with the image and select
    the most semantically similar one.

    Args:
        image: PIL Image to generate captions for
        model: Model to use for inference
        max_length: Maximum length of generated captions

    Returns:
        CaptionResponse containing the best caption and all generated captions with their details
    """
    caption_details = sample_five_inference(image, model, max_length)
    captions = [detail.caption for detail in caption_details]

    # Filter out captions that would exceed CLIP's token limit
    valid_captions = []
    valid_indices = []

    for idx, caption in enumerate(captions):
        tokens = CLIP_PROCESSOR.tokenizer(
            caption, return_tensors="pt", truncation=False
        )
        if len(tokens.input_ids[0]) <= 77:  # CLIP's max token length
            valid_captions.append(caption)
            valid_indices.append(idx)

    if not valid_captions:
        # If no valid captions, return the shortest one
        shortest_idx = min(range(len(captions)), key=lambda i: len(captions[i]))
        return CaptionResponse(
            caption=captions[shortest_idx], all_captions=caption_details
        )

    # Process image and valid captions with CLIP
    inputs = CLIP_PROCESSOR(
        text=valid_captions,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # Get similarity scores
    with torch.no_grad():
        outputs = CLIP_MODEL(**inputs)
        logits_per_image = outputs.logits_per_image

    # Get index of most similar caption among valid ones
    best_valid_idx = logits_per_image.argmax().item()
    best_idx = valid_indices[best_valid_idx]

    print(f"\nSelected caption {best_idx + 1} as the best match")
    print(f"All logits: {logits_per_image}")
    print(f"Winner: {captions[best_idx]}")
    print("-" * 100)

    return CaptionResponse(caption=captions[best_idx], all_captions=caption_details)


def eval_sample():
    """Evaluate the model on a random training image."""
    dataset = Flickr30kDataset(split="test")

    for i in range(10):
        image, gt_caption = dataset[i]

        # Get best caption
        caption, _, _ = get_best_caption(image)
        display_image_with_caption(image, caption, gt_caption)


def eval_from_url(url: str):
    """Evaluate the model on an image from a URL."""
    image = Image.open(requests.get(url, stream=True).raw)
    caption, _, _ = get_best_caption(image)
    display_image_with_caption(image, caption)
