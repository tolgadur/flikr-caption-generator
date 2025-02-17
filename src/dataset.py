import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        dataset = load_dataset("nlphuji/flickr30k", split="test")
        self.ds = dataset.filter(lambda x: x["split"] == split)

        # load the model and processor to get the image and text embeddings
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get the image and text
        image = self.ds[idx]["image"]
        captions = self.ds[idx]["caption"]

        # get the image and text embeddings
        inputs = self.processor(
            images=image, text=captions, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        return image_embeds, text_embeds
