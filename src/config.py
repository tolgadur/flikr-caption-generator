import torch
from model import FlickrImageCaptioning
from transformers import CLIPProcessor, CLIPModel


def load_model(model_path: str = "models/model.pth"):
    """Load the model from checkpoint and set to eval mode."""
    model = FlickrImageCaptioning().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize models globally
torch.manual_seed(42)
MODEL = load_model()
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set CLIP model to eval mode
CLIP_MODEL.eval()
