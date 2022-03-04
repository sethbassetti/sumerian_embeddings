import os
import torch
from sumerian_data import SumerianDataset, SumerianParallelDataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from transformers import T5Tokenizer
from utils import train_fast_text_embeddings
import sentencepiece as spm

def main():

    
    # Creates the fast text embeddings if they don't already exist
    if not os.path.isfile('models/sum_embeddings/fasttext.model'):
        if not os.path.isdir('models/sum_embeddings'):
            os.mkdir('models/sum_embeddings')
        train_fast_text_embeddings()
    model = Word2Vec.load('models/sum_embeddings/fasttext.model')
    print(model.wv.most_similar('masz2'))
    # masz2 is goat and lugal is king, to get similarity between goat and sheep, change 'lugal' to 'udu' or 'u8'
    print(model.wv.similarity('masz2', 'lugal'))

    # The sumerian tokenizer, using a sentencepiece model
    sum_tokenizer = spm.SentencePieceProcessor(model_file='models/sum_tokenizer/m.model')
    print(sum_tokenizer.encode('gar lugal udu masz2 gar', add_bos=True))

    # The english tokenizer, using a T5 Tokenizer model
    eng_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    print(eng_tokenizer("The dog walks in the park", return_tensors="pt").input_ids)
    

if __name__ == "__main__":
    main()