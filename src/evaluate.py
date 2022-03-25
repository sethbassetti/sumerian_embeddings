

from os import join, isfile, dirname, abspath
from sumerian_model import SumerianParallelDataset
from sumerian_model import construct_model
from utils import translate_sentence
from transformers import T5Tokenizer
import torch
import sentencepiece as spm
import sacrebleu

def main():
    # Sets up the paths to the various data/models/checkpoint the script needs
    root_path = dirname(dirname(abspath(__file__)))
    data_path = join(root_path, 'data')
    model_path = join(root_path, 'models')
    checkpoint_path = join(root_path, 'checkpoints', 't5_checkpoint.pt')

    device = 'cuda'


    model = construct_model.eval()

    # Loads the checkpoint if it exists
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['mdl_state_dict'])

    # Initializes test set
    test_set = SumerianParallelDataset(english_corpus=join(data_path, 'eng_lines_test.txt'), sumerian_corpus=join(data_path, 'sum_lines_test.txt'))
    sum_lines = [sumerian for (sumerian, _) in test_set]
    eng_lines = [english for (_, english) in test_set]

    # Constructs sumerian and english tokenizers
    sum_tokenizer = T5Tokenizer.from_pretrained(join(model_path, 'sum_tokenizer', 'm.model'), sp_model = spm.SentencePieceProcessor(model_file=join(model_path, 'sum_tokenizer', 'm.model')))
    eng_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    
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



if __name__ == "__main__":
    main()