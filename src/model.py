import torch
import torch.nn as nn
from transformers import CLIPModel

from decoder import Decoder
from config import VOCAB_SIZE


class FlickrImageCaptioning(nn.Module):
    def __init__(self, d_model=512, heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        """
        End-to-end model for image captioning.
        Uses CLIP embeddings and a transformer decoder.
        The model has to have d_model of 512 as this is used in the CLIP model
        from the transformers library.
        """
        # CLIP components
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = self.clip_model.vision_model
        self.text_model = self.clip_model.text_model

        # Decoder
        self.decoder = Decoder(d_model, heads, n_layers, dropout)

        # Output projection
        self.classifier = nn.Linear(d_model, VOCAB_SIZE)

    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process images through CLIP vision model."""
        img_out = self.vision_model(pixel_values=pixel_values)
        img_embeds = self.clip_model.visual_projection(img_out.pooler_output)
        return img_embeds.unsqueeze(1)  # [batch_size, 1, d_model]

    def get_text_embeddings(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Process text through CLIP text model."""
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=mask)
        return self.clip_model.text_projection(text_outputs.last_hidden_state)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            pixel_values: Image pixel values [batch_size, 3, 224, 224]
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        # Get image embeddings
        img_embeds = self.get_image_embeddings(pixel_values)

        # Get text embeddings
        text_embeds = self.get_text_embeddings(input_ids, mask)

        # Combine image and text embeddings
        x = torch.cat((img_embeds, text_embeds), dim=1)

        # Pass through decoder
        x = self.decoder(x)

        # Project to vocabulary
        return self.classifier(x)
