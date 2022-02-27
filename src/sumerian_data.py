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

        # Processes the text and splits it into a list of words
        processed_doc_text = self.process_text(doc_text)
        processed_doc_text = processed_doc_text.split()

        return processed_doc_text

    def process_text(self, text):
        """Performs a number of regular expression substitutions to sanitize the text so that
        all words/signs exist within the vocabulary list

        Args:
            text (str): The raw text existing on the document. It can include any side/edge of the document

        Returns:
            str: The text that has been processed according to a number of substitutions.
        """
        # Substitutes a number of uneccesary characters out of the text.
        # The # character indicates that the sign it follows was damaged
        # The [] characters indicate that the sign inside the brackets was broken off
        # The <> characters indicate that there was an accidental omission via the editor
        # The pipe "|" character indicates a compound grapheme (a sign within another sign/ on top of, etc...)
        # The "?" character indicates an uncertainty of the identification of the sign
        # The "!" character indicates there was a correction of the sign
        # None of these characters exist in the vocab so they can be removed
        text = re.sub(r'[#\[\]<>|?!]', '', text)
        
        # This removes any occurence of "($ 'some text' $)" since these are generally inline comments such as
        # "blank space"
        text = re.sub(r'\(\$.*\$\)', '', text)
        
        # This removes any occurence of a stand-alone ellipses "..." at the beginning of the text or elsewhere
        # These ellipses indicate that an undeterminable number of signs may be missing from the tablet
        text = re.sub(r'(\s|^)(\.\.\.\s)+', ' ', text)

        # If there is an underscore found in the text, then return an empty string, invalidating this document.
        # Underscores are used to represent logograms in non-sumerian languages. So if an underscore is found,
        # there must have been a mistake with the language labelling, so just throw out this sample
        if "_" in text:
            return ""

        # Sometimes floating syllabograms can appear as a result of certain substitutions. An example is
        # 1/2(disz) -ta, where they should be connected. This finds all instances of a "floating syllabogram"
        # with a leading hyphen and connects it back to the original word
        text = re.sub(r'\s+-', '-', text)

        # The colon character by itself represents a sumerian punctuation mark and is not needed for our tasks
        text = re.sub(r'\s:\s', ' ', text)

        #NOTE: I could not figure out why the next three occurences appear within the dataset, more research
        # is needed

        # Removes any occurence of a standalone number, not attached to any sign
        text = re.sub(r'\s\d+\s', ' ', text)

        # Removes any sign that starts with a number and then a hyphen
        text = re.sub('\s\d-.*\s', ' ', text)

        # Removes any occurence of a standalone ampersands
        text = re.sub('\s&\s', ' ', text)

        return text
                

    def in_vocab(self, word):
        return word in self.vocab
    
    def __len__(self):
        """Returns the length of the dataset"""

        return len(self.sentences)

    def __getitem__(self, idx: int):
        """Retrieves a single item from the dataset"""

        return self.sentences[idx]