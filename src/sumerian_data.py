import torch
from torch.utils.data import Dataset

class SumerianDataset(Dataset):
    """The dataset class that contains the sumerian texts"""

    def __init__(self, corpus_dir):
        """ Initializes the dataset and loads the corpus into memory"""

        self.sentences = self.load_corpus(corpus_dir)

    def load_corpus(self, corpus_dir):
        """Loads the sentences from the corpus into the dataset class

        Args:
            corpus_dir (string): Filepath to the corpus

        Returns:
            list[str]: A list of sumerian text sentences
        """

        with open(corpus_dir, "r") as corpus:
            sentences = []

            # Strips each line in the corpus and appends it to the list of sentences
            for line in corpus.readlines():
                sentences.append(line.strip())

        return sentences
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]