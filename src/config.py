import torch
import os
from model import FlickrImageCaptioning
from transformers import CLIPProcessor, CLIPModel


def load_model(
    model_path: str = os.path.join(os.getenv("MODEL_PATH", "models"), "model.pth"),
    clip_model=None,
):
    """Load the model from checkpoint and set to eval mode."""
    print("Loading FlickrImageCaptioning model...")
    model = FlickrImageCaptioning(clip_model=clip_model).to(DEVICE)
    print(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully")
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize models globally
torch.manual_seed(42)
print("Starting to initialize models...")

# Load CLIP model first
print("Loading CLIP model...")
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
print("Loading CLIP processor...")
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FlickrImageCaptioning model with shared CLIP
MODEL = load_model(clip_model=CLIP_MODEL)

# Set CLIP model to eval mode
CLIP_MODEL.eval()
print("All models initialized and ready")
