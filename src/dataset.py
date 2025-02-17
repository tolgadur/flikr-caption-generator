import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset("nlphuji/flickr30k", split="test").select(range(100))
        self.ds = dataset.filter(lambda x: x["split"] == split)

        # load the model and processor to get the image and text embeddings
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = model.text_model
        self.vision_model = model.vision_model

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get the image and text
        image = self.ds[idx]["image"]
        captions = self.ds[idx]["caption"]

        # get the image and text embeddings
        image_inputs = self.processor(images=image, return_tensors="pt")
        text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)

        image_outputs = self.vision_model(**image_inputs)
        text_outputs = self.text_model(**text_inputs)

        image_embeds = image_outputs.pooler_output  # shape: [1, 768]
        text_embeds = text_outputs.last_hidden_state  # shape: [5, 22, 512]

        return image_embeds, text_embeds
