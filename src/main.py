#! /usr/bin/env python3

import os
from os.path import isfile, dirname, abspath, join
import wandb
import torch
from sumerian_data import SumerianDataset, SumerianParallelDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from gensim.models import Word2Vec
from transformers import T5Tokenizer
from utils import train_fast_text_embeddings
from sumerian_model import construct_model
import sentencepiece as spm
import hydra

import nltk
import time

def setup(rank, world_size):
    """This function sets up the environment for multiprocessing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



@hydra.main(config_path='../config', config_name='default')
def launch_processes(cfg):
    """ Launches multiple GPU processes"""

    # Gets num of available GPU devices
    world_size = torch.cuda.device_count()

    print(f"Setting up processes on {world_size} GPUs")

    # Spawns world_size processes with the config 
    mp.spawn(train,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)



def train(rank, world_size, cfg):

    # Sets up the process so all processes can communicate on same host
    setup(rank, world_size)

    # Reads the config to obtain hyperparameters
    hyperparameters = cfg['hyperparameters']
    epochs = hyperparameters['epochs']
    lr = hyperparameters['lr']
    max_source_length = hyperparameters['max_source_length']
    max_target_length = hyperparameters['max_target_length']

    # Sets up the paths to the various data/models/checkpoint the script needs
    root_path = dirname(dirname(abspath(__file__)))
    data_path = join(root_path, 'data')
    model_path = join(root_path, 'models')
    checkpoint_path = join(root_path, 'checkpoints', 't5_checkpoint.pt')
    
    # Constructs sumerian and english tokenizers
    sum_tokenizer = T5Tokenizer.from_pretrained(join(model_path, 'sum_tokenizer', 'm.model'), sp_model = spm.SentencePieceProcessor(model_file=join(model_path, 'sum_tokenizer', 'm.model')))
    eng_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # This sets the device to the appropriate GPU
    torch.cuda.set_device(rank)

    # Initializes the model and optimizer
    model = construct_model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initializes training set and dataloader
    train_set = SumerianParallelDataset(english_corpus=join(data_path, 'eng_lines_train.txt'), sumerian_corpus=join(data_path, 'sum_lines_train.txt'))

    # Sampler that takes care of the distribution of the batches such that
    # the data is not repeated in the iteration and sampled accordingly
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_set,
    	num_replicas=world_size,
    	rank=rank
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

    # Loads the checkpoint if it exists
    if isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['mdl_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if rank == 0:
        wandb.init()

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
            outputs = model(input_ids.cuda(rank), labels=labels.cuda(rank), attention_mask=attention_mask.cuda(rank))

            # Retrieves the loss from the model
            loss = outputs.loss

            # Send the loss back through the model and update the parameters
            loss.backward()
            optimizer.step()

            running_loss += loss

        # Only print this out for 1 gpu
        if rank == 0:
            epoch_loss = running_loss / len(train_loader)
            # Every epoch, log  and print the loss
            wandb.log({'loss': epoch_loss})
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")

    if rank == 0:
        torch.save({'epoch': epoch,
                    'mdl_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss},
                    checkpoint_path)

"""               # Every epoch save a checkpoint
                torch.save({'epoch': epoch,
                            'mdl_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss},
                            checkpoint_path)
                            """

"""
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
        """


def test():

    T5Tokenizer.from_pretrained('models/sum_tokenizer/')
    
if __name__ == "__main__":
    launch_processes()
