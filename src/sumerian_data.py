from torch.utils.data import Dataset
from typing import List
import re

NEW_DOC="&"

class SumerianDataset(Dataset):
    """The dataset class that contains the sumerian texts. The dataset performs all preprocessing of the text
    on initialization and loads the text into memory for retrieval by a dataloader."""

    def __init__(self, corpus_dir: str,):
        """ Initializes the dataset and loads the corpus into memory"""

        self.sentences: List[str] = self.load_corpus(corpus_dir)


    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> List[str]:
        return self.vocab
    
    def get_corpus(self) -> List[str]:
        return self.sentences

    def load_corpus(self, corpus_dir: str) -> List[str]:
        """Loads the sentences from the corpus into the dataset class

        Args:
            corpus_dir (string): Filepath to the corpus

        Returns:
            list[str]: A list of sumerian text sentences
        """
        
        # A list that contains each document in the corpus (tablet, seal, envelope, etc...)
        with open(corpus_dir, 'r') as corpus:
            documents = [line.strip().split() for line in corpus.readlines()]
      
        return documents          
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]

class SumerianParallelDataset(Dataset):
    """The dataset class that contains sumerian sentences and english translations. 
    The dataset retrieves the parallel data from disk and loads it into memory to be used in the dataloader."""

    def __init__(self, english_corpus=None, sumerian_corpus=None):
        """ Initializes the dataset and loads the corpus into memory"""

        # Verifies that a corpus is passed in
        if english_corpus is None or sumerian_corpus is None:
            raise Exception("You must pass an english and a sumerian corpus into dataset")

        # Loads the data into the sentences
        self.eng_sents, self.sum_sents = self.load_parallel_data(english_corpus, sumerian_corpus)

    def load_parallel_data(self, english_corpus, sumerian_corpus) -> List[str]:
        """Loads the sentences from the corpus into the dataset class

        Args:
            corpus_dir (string): Filepath to the corpus

        Returns:
            list[str]: A list of sumerian text sentences
        """
        
        # A file containing a list of english translations
        with open(english_corpus, 'r') as corpus:
            eng_lines = [line.strip() for line in corpus.readlines()]
        
        with open(sumerian_corpus, 'r') as corpus:
            sum_lines = [line.strip() for line in corpus.readlines()]

      
        return eng_lines, sum_lines
                
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.eng_sents)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sum_sents[idx], self.eng_sents[idx]