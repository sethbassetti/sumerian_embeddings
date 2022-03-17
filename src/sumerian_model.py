from transformers import T5ForConditionalGeneration
import torch

def construct_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    #This switches the encoder embedding to work with our tokenizer for sumerian
    model.encoder.embed_tokens = torch.nn.Embedding(1000, 512)

    model = model.train()
    return model

def main():
    pass

if __name__ == "__main__":
    main()