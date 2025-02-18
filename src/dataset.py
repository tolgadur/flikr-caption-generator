import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset("nlphuji/flickr30k", split="test")
        self.ds = dataset.filter(lambda x: x["split"] == split)

        # load the model and processor to get the image and text embeddings
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = self.model.text_model
        self.vision_model = self.model.vision_model

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get the image and text
        image = self.ds[idx]["image"]
        captions = self.ds[idx]["caption"]

        # add start/end tokens to each caption as string
        bos_token = self.processor.tokenizer.bos_token
        eos_token = self.processor.tokenizer.eos_token

        # create input sequences (BOS + caption) and target sequences (caption + EOS)
        input_caps = [f"{bos_token} {caption}" for caption in captions]
        target_caps = [f"{caption} {eos_token}" for caption in captions]

        # get the image and text embeddings
        image_inputs = self.processor(images=image, return_tensors="pt")
        input_seq = self.processor(text=input_caps, return_tensors="pt", padding=True)
        target_seq = self.processor(text=target_caps, return_tensors="pt", padding=True)

        image_outputs = self.vision_model(**image_inputs)
        input_outputs = self.text_model(**input_seq)
        target_outputs = self.text_model(**target_seq)

        image_embeds = self.model.visual_projection(
            image_outputs.pooler_output
        )  # shape: [1, 512] - CLS token
        input_embeds = self.model.text_projection(
            input_outputs.last_hidden_state
        )  # shape: [5, 25, 512]
        target_embeds = self.model.text_projection(
            target_outputs.last_hidden_state
        )  # shape: [5, 25, 512]

        image_as_tensor = image_inputs["pixel_values"].squeeze(0).permute(1, 2, 0)

        return (
            image_as_tensor,
            image_embeds,
            input_embeds,
            target_embeds,
        )
