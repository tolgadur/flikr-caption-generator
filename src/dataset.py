import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset


class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", seed=42):
        torch.manual_seed(seed)
        ds = load_dataset("nlphuji/flickr30k", split="test").select(range(100))
        self.ds = ds.filter(lambda x: x["split"] == split)

        self.pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = self.clip_model.vision_model
        self.text_model = self.clip_model.text_model

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        itm = self.ds[idx]
        img = itm["image"]
        txt = itm["caption"][0]  # todo: make this smarter

        enc = self.pr(images=img, text=txt, return_tensors="pt")
        tkn = enc["input_ids"]  # shape [1, seq_len + 2] for bos and eos
        img = enc["pixel_values"]  # shape [1, 3, 224, 224]
        mask = enc["attention_mask"][:, 1:]  # shape [1, seq_len + 1]

        inp_eq = tkn[:, 1:]  # shape [1, seq_len + 1]
        tgt_eq = tkn[:, :-1]  # shape [1, seq_len + 1]

        img_out = self.vision_model(pixel_values=img)  # shape [1, 768]
        inp_out = self.text_model(input_ids=inp_eq, attention_mask=mask)
        tgt_out = self.text_model(input_ids=tgt_eq, attention_mask=mask)

        # linear projection to match model dimension 512 and use CLS token
        img_embeds = self.clip_model.visual_projection(img_out.pooler_output)
        inp_embeds = self.clip_model.text_projection(inp_out.last_hidden_state)
        tgt_embeds = self.clip_model.text_projection(tgt_out.last_hidden_state)

        img_plt = img.squeeze(0).permute(1, 2, 0)  # shape [224, 224, 3]
        return (img_plt, img_embeds, inp_embeds, tgt_embeds)
