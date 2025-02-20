from pydantic import BaseModel, HttpUrl
from typing import List


class ImageUrl(BaseModel):
    url: HttpUrl


class CaptionDetails(BaseModel):
    caption: str
    temperature: float
    is_argmax: bool
    model_type: str  # 'multimodal' or 'single'


class CaptionResponse(BaseModel):
    caption: str  # The best/selected caption
    all_captions: List[CaptionDetails]  # All generated captions with their details
