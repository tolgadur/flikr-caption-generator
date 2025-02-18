import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
from PIL.Image import Image


class CLIPEmbedder:
    def __init__(self):
        self.pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = self.clip_model.vision_model
        self.text_model = self.clip_model.text_model

    def __call__(self, imgs: List[Image], txts: List[str]) -> Tuple[torch.Tensor, ...]:
        # Process the batch with padding
        enc = self.pr(images=imgs, text=txts, return_tensors="pt", padding=True)

        # Get tokenized inputs
        tkn = enc["input_ids"]  # shape [batch_size, seq_len + 2] for bos and eos
        img = enc["pixel_values"]  # shape [batch_size, 3, 224, 224]
        mask = enc["attention_mask"][:, 1:]  # shape [batch_size, seq_len + 1]

        # Create input and target sequences for training
        inp_eq = tkn[:, 1:]  # shape [batch_size, seq_len + 1]
        tgt_eq = tkn[:, :-1]  # shape [batch_size, seq_len + 1]

        # Get embeddings
        img_out = self.vision_model(pixel_values=img)
        inp_out = self.text_model(input_ids=inp_eq, attention_mask=mask)
        tgt_out = self.text_model(input_ids=tgt_eq, attention_mask=mask)

        # Project to the same model dimension of 512
        img_embeds = self.clip_model.visual_projection(img_out.pooler_output)
        inp_embeds = self.clip_model.text_projection(inp_out.last_hidden_state)
        tgt_embeds = self.clip_model.text_projection(tgt_out.last_hidden_state)

        # Convert images to the format expected by the model
        img_plt = img.permute(0, 2, 3, 1)  # shape [batch_size, 224, 224, 3]

        return img_plt, img_embeds, inp_embeds, tgt_embeds


def collate_fn(
    batch: List[Tuple[Image, str]], embedder: CLIPEmbedder
) -> Tuple[torch.Tensor, ...]:
    """
    Collate function to be used with DataLoader.
    Args:
        batch: List of tuples (image, text)
        embedder: Instance of CLIPEmbedder
    Returns:
        Tuple of tensors (images_plt, image_embeds, input_embeds, target_embeds)
    """
    images, texts = zip(*batch)
    return embedder(list(images), list(texts))
