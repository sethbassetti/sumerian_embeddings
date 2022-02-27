from distutils import text_file
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

        vocab = {}
        with open(vocab_dir, 'r') as vocab_file:
            for word in vocab_file.readlines():
                vocab[word.strip()] = 0
        return vocab

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

            # Contains information about each document 
            document = []

            # Iterates through the dataset, appending documents to the list
            for line in corpus.readlines():

                # This indicates the start of a document
                if line.startswith(NEW_DOC):
                   
                    # If the document isn't empty, processes the document into the raw transliterated text
                    if document:
                        processed_document = self.process_document(document)
                    else:
                        continue

                    # If the processed document is valid, then add it to the documents list
                    if processed_document is not None:
                        documents.append(processed_document)

                    # Reset the document to an empty list
                    document = []

                # Otherwise create the document
                else:
                    document.append(line.strip())



        return documents

    def process_document(self, document):
        """Takes the raw document data in ATF format and converts it into a tokenized sequence of sumerian words

        Args:
            document (List[str]): A list of lines containing document information

        Returns:
            List[str]: A list of tokens that have been processed or the empty string if the document is invalid
        """

        doc_text = ""
        
        # Performs a regex to retrieve the language of this document
        match = re.compile('#atf: lang(.*)').match(document[0])

        # If the above regex doesn't match, this is not a regular document
        if match:

            # Only continue if the language is "sux", which is Sumerian
            if match.group(1).strip() == 'sux':
                for line in document:

                    # Make sure that the line is not a comment line
                    if not line.startswith("#"):
                        # This regex checks for a sequence of non-space characters, followed by a period then a space
                        # This matches all text lines in the document
                        text_line_match = re.compile('\S+\.\s(.*)').match(line)
                        if text_line_match:

                            # Assigns the sumerian transliteration text found in the line
                            text = text_line_match.group(1)
                            doc_text += text + " "

        processed_doc_text = self.process_text(doc_text)
        processed_doc_text = processed_doc_text.split()

        return processed_doc_text

    def process_text(self, text):
        
        orig_text = text
        text = text.replace("#", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("<", "")
        text = text.replace(">", "")
        text = text.replace("|", "")
        text = text.replace("?", "")
        text = text.replace("!", "")
        #text = text.replace("_", "")
        
        text = re.sub(r'\(\$.*\$\)', '', text)
        
        
        while re.compile(r'\s\.\.\.\s').search(text):
            text = re.sub(r'\s\.\.\.\s', " ", text)
        text = re.sub(r'^\.\.\.\s', " ", text)
        #text = text.replace(" x- ", "")

        if "_" in text:
            return ""

        if " -ta " in text:
            breakpoint()

        return text
                

    def in_vocab(self, word):
        return word in self.vocab
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]