from torch.utils.data import Dataset
from typing import List
import re

NEW_DOC="&"

class SumerianDataset(Dataset):
    """The dataset class that contains the sumerian texts. The dataset performs all preprocessing of the text
    on initialization and loads the text into memory for retrieval by a dataloader."""

    def __init__(self, corpus_dir: str, vocab_dir: str):
        """ Initializes the dataset and loads the corpus into memory"""

        self.vocab: List[str] = self.load_vocab(vocab_dir)

        self.sentences: List[str] = self.load_corpus(corpus_dir)

    def load_vocab(self, vocab_dir: str) -> List[str]:
        """Using the path to the vocab list, constructs a list of words representing the sumerian vocabulary

        Args:
            vocab_dir (str): A path to the vocab file

        Returns:
            List[str]: A list of sumerian vocab words
        """
        
        with open(vocab_dir, 'r') as vocab_file:

            # Strips each word to remove the newline character
            vocab = [word.strip() for word in vocab_file.readlines()]
        return vocab

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
                

    def in_vocab(self, word):
        return word in self.vocab
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]