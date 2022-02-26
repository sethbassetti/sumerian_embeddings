import torch
from torch.utils.data import Dataset
from typing import List
import re

SUM_LANG="#atf: lang sux"
NEW_DOC="&"

class SumerianDataset(Dataset):
    """The dataset class that contains the sumerian texts"""

    def __init__(self, corpus_dir: str, vocab_dir: str):
        """ Initializes the dataset and loads the corpus into memory"""

        self.vocab: List[str] = self.load_vocab(vocab_dir)

        self.sentences: List[str] = self.load_corpus(corpus_dir)

    def load_vocab(self, vocab_dir: str) -> List[str]:
        pass

    def load_corpus(self, corpus_dir: str) -> List[str]:
        """Loads the sentences from the corpus into the dataset class

        Args:
            corpus_dir (string): Filepath to the corpus

        Returns:
            list[str]: A list of sumerian text sentences
        """
        
        # A list that contains each document in the corpus (tablet, seal, envelope, etc...)
        documents = []

        with open(corpus_dir, "r") as corpus:

            is_sux = False
            
            document = []
            # Iterates through the dataset, appending documents to the list
            for line in corpus.readlines():

                # This indicates the start of a document
                if line.startswith(NEW_DOC):

                    # Processes the document into the raw transliterated text
                    processed_document = self.process_document(document)

                    documents.append(processed_document)
                    document = []
                    is_sux = False

                elif SUM_LANG in line:
                    is_sux = True

                elif is_sux:
                    document.append(line)



        return documents

    def process_document(self, document):
        for line in document:
            language = re.compile("#atf: lang sux").match(line)
            
                


    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]