#! /usr/bin/env python3

from os.path import isfile, dirname, abspath, join
import wandb
import torch
from sumerian_data import SumerianDataset, SumerianParallelDataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from transformers import T5Tokenizer
from utils import train_fast_text_embeddings
from sumerian_model import construct_model
import sentencepiece as spm
import hydra
import sacrebleu
import nltk
import time

def translate_sentence(sumerian, sum_tokenizer, eng_tokenizer, model, device):
    """ Utility function to translate a sentence from sumerian - english"""

    input_ids = sum_tokenizer(sumerian, return_tensors='pt', max_length=512, padding='longest', truncation=True).input_ids
    output = model.generate(input_ids.to(device))

    decoded_output = eng_tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output


@hydra.main(config_path='../config', config_name='default')
def main(cfg):

    # Reads the config to obtain hyperparameters
    hyperparameters = cfg['hyperparameters']
    epochs = hyperparameters['epochs']
    lr = hyperparameters['lr']
    max_source_length = hyperparameters['max_source_length']
    max_target_length = hyperparameters['max_target_length']
    device = hyperparameters['device']
    mode = hyperparameters['mode']

    # Sets up the paths to the various data/models/checkpoint the script needs
    root_path = dirname(dirname(abspath(__file__)))
    data_path = join(root_path, 'data')
    model_path = join(root_path, 'models')
    checkpoint_path = join(root_path, 'checkpoints', 't5_checkpoint.pt')
    
    
    # Constructs sumerian and english tokenizers
    sum_tokenizer = T5Tokenizer.from_pretrained(join(model_path, 'sum_tokenizer', 'm.model'), sp_model = spm.SentencePieceProcessor(model_file=join(model_path, 'sum_tokenizer', 'm.model')))
    eng_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Initializes the model and optimizer
    model = construct_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initializes training set and dataloader
    train_set = SumerianParallelDataset(english_corpus=join(data_path, 'eng_lines_train.txt'), sumerian_corpus=join(data_path, 'sum_lines_train.txt'))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=3, pin_memory=True)

    # Loads the checkpoint if it exists
    if isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['mdl_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if mode == 'train':

        # Puts the model in training mode
        model.train()
        # Initialize a wandb project
        wandb.init(project='sumerian_embeddings')
        
        for epoch in range(epochs):
            running_loss = 0

            for _, data in enumerate(train_loader):

                # All the data will be input/label pairs
                inputs, labels = data

                # Encodes the input nnto sumerian tokens
                input_encoding = sum_tokenizer(inputs, return_tensors='pt', max_length=max_source_length, padding='longest', truncation=True)
                input_ids = input_encoding.input_ids
                attention_mask = input_encoding.attention_mask

                # Tokenizes the english label
                labels = eng_tokenizer(labels, return_tensors='pt', max_length=max_target_length, padding='longest', truncation=True).input_ids

                labels[labels == eng_tokenizer.pad_token_id] = -100


                # Zero out the gradients before every batch is passed through
                optimizer.zero_grad()

                # Send the data through the model
                outputs = model(input_ids.to(device), labels=labels.to(device), attention_mask=attention_mask.to(device))

                # Retrieves the loss from the model
                loss = outputs.loss

                # Send the loss back through the model and update the parameters
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            # Every epoch, log  and print the loss
            wandb.log({'loss': epoch_loss})
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")

            # Every epoch save a checkpoint
            torch.save({'epoch': epoch,
                        'mdl_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss},
                        checkpoint_path)

    elif mode=='eval':
        model.eval()

        # Initializes test set
        test_set = SumerianParallelDataset(english_corpus=join(data_path, 'eng_lines_test.txt'), sumerian_corpus=join(data_path, 'sum_lines_test.txt'))
        sum_lines = [sumerian for (sumerian, english) in test_set]
        eng_lines = [english for (sumerian, english) in test_set]

        
        if not isfile(join(data_path, 'test_translations.txt')):
            hypotheses = [translate_sentence(sentence, sum_tokenizer, eng_tokenizer, model, device) for sentence in sum_lines]
            with open(join(data_path, 'test_translations.txt'), 'w') as outfile:
                hypotheses = [sentence + '\n' for sentence in hypotheses]
                outfile.writelines(hypotheses)

        with open(join(data_path, 'test_translations.txt'), 'r') as infile:
            hypotheses = [line.strip() for line in infile.readlines()]


        references = [line for line in eng_lines]

        # Computes the BLEU score on the dataset
        print(sacrebleu.corpus_bleu(hypotheses, references))


def test():

    T5Tokenizer.from_pretrained('models/sum_tokenizer/')
    
if __name__ == "__main__":
    main()
   # test()