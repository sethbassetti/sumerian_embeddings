import wandb
import torch
from sumerian_data import SumerianDataset, SumerianParallelDataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from transformers import T5Tokenizer
from utils import train_fast_text_embeddings
from sumerian_model import construct_model
import sentencepiece as spm

def main():
    wandb.init(project='sumerian_embeddings')

    # Constructs sumerian and english tokenizers
    sum_tokenizer = T5Tokenizer.from_pretrained('models/sum_tokenizer/m.model', sp_model = spm.SentencePieceProcessor(model_file='models/sum_tokenizer/m.model'))
    eng_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    model = construct_model()
    
    # Defining Hyperparameters
    epochs = 1
    lr = 1e-4
    max_source_length = 512
    max_target_length = 128

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_set = SumerianParallelDataset(english_corpus='data/eng_lines.txt', sumerian_corpus='data/sum_lines.txt')
    train_loader = DataLoader(train_set, batch_size=64)

    for epoch in range(epochs):
        running_loss = 0

        for i, data in enumerate(train_loader):

            # All the data will be input/label pairs
            inputs, labels = data

            input_encoding = sum_tokenizer(inputs, return_tensors='pt', max_length=max_source_length, padding='longest', truncation=True)
            input_ids = input_encoding.input_ids
            attention_mask = input_encoding.attention_mask
            labels = eng_tokenizer(labels, return_tensors='pt', max_length=max_target_length, padding='longest', truncation=True).input_ids

            labels[labels == eng_tokenizer.pad_token_id] = -100


            # Zero out the gradients before every batch is passed through
            optimizer.zero_grad()
  
            # Send the data through the model
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            # Retrieves the loss from the model
            loss = outputs.loss

            # Send the loss back through the model and update the parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Every 500 datapoints, print the running loss
            wandb.log({'loss': (running_loss / (i+1))})
            print(f"Epoch {epoch}. Example {i}. Loss: {running_loss / (i+1)}")


def test():
    breakpoint()
    T5Tokenizer.from_pretrained('models/sum_tokenizer/')
    
if __name__ == "__main__":
    main()
   # test()