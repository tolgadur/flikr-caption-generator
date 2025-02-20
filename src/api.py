from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import requests
from typing import Dict, List
from pydantic import BaseModel, HttpUrl

from config import MODEL
from evals import get_best_caption

router = APIRouter()


class ImageUrl(BaseModel):
    url: HttpUrl


async def process_image_and_generate_caption(img: Image.Image) -> str:
    """Helper function to generate caption from PIL Image."""
    try:
        return get_best_caption(img, model=MODEL)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating caption: {str(e)}"
        )


@router.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/generate-caption")
async def generate_caption(image: UploadFile = File(...)) -> Dict[str, str | List[str]]:
    """
    Generate a caption for the uploaded image.

    Args:
        image: The image file to generate a caption for

    Returns:
        Dict containing the generated caption and list of all captions
    """
    try:
        # Read and convert the uploaded file to PIL Image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    best_cap, all_caps = await process_image_and_generate_caption(img)
    return {
        "caption": best_cap,
        "all_captions": all_caps,
    }


@router.post("/generate-caption-from-url")
async def generate_caption_from_url(image_url: ImageUrl) -> Dict[str, str | List[str]]:
    """
    Generate a caption for an image from a URL.

    Args:
        image_url: The URL of the image to generate a caption for

    Returns:
        Dict containing the generated caption and list of all captions
    """
    try:
        response = requests.get(str(image_url.url), stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Error fetching image from URL: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    best_cap, all_caps = await process_image_and_generate_caption(img)
    return {
        "caption": best_cap,
        "all_captions": all_caps,
    }
