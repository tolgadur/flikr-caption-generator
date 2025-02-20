from transformers import CLIPProcessor


def get_vocab():
    pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    word2index = pr.tokenizer.get_vocab()
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word


TOKEN2INDEX, INDEX2TOKEN = get_vocab()
VOCAB_SIZE = len(TOKEN2INDEX)
