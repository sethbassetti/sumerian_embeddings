import sentencepiece as spm
from gensim.models import FastText

def train_sum_tokenizer():

    # Trains a sentence piece tokenizer to construct a vocab and model for the sumerian text
    spm.SentencePieceTrainer.train(input='data/sum_lines.txt', model_prefix='m', vocab_size=1000, eos_id=1, bos_id=0, unk_id=2)

def train_fast_text_embeddings():
    """Trains a Fast Text Model on the sumerian dataset to construct word embeddings"""

    # Opens up the sumerian text and creates a list of tokens
    with open('data/clean_sumerian.txt', 'r') as corpus:
        sentences = [sentence.strip().split() for sentence in corpus.readlines()]

    # Uses those tokens to create a fast text embedding model of length-100 embedding vectors
    # Window is the context window it uses, min count is the num of times a word needs to appear to include it
    # workers just makes it run quicker
    model = FastText(sentences=sentences, vector_size=100, window=10, min_count=2, workers=4)
    model.save('models/sum_embeddings/fasttext.model')


def main():
    train_sum_tokenizer()
    #train_fast_text_emeddings()

if __name__ == '__main__':
    main()