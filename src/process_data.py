import re
from pathlib import Path

NEW_DOC_SYMB = '&'

def process_data(input_path, output_path):

    # This will be a list of the raw ATF documents that we will process
    raw_documents = []

    with open(input_path, "r") as corpus:

            # Contains information about each document 
            document = ""

            # Iterates through the dataset, appending documents to the list
            for line in corpus.readlines():

                # Adds on to the document until a NEW DOC SYMB is encountered, then adds completed document
                # to raw_documents list and repeats
                if line.startswith(NEW_DOC_SYMB) and document:
                   raw_documents.append(document)
                   document = line
                else:
                    document += line

    # Filters the documents to only get sumerian languages
    filtered_docs = filter_documents(raw_documents)
    processed_docs = process_documents(filtered_docs)
    write_documents(processed_docs, output_path)

    # Extracts all lines that have been translated and creates a parallel dataset with them
    sum_lines, eng_lines = extract_translations(filtered_docs)
    write_documents(sum_lines, 'sum_lines.txt')
    write_documents(eng_lines, 'eng_lines.txt')
    
    # Splits both the datasets into train/dev/test splits
    split_dataset('data/sum_lines.txt')
    split_dataset('data/eng_lines.txt')

def filter_documents(raw_documents):
    """Filters out all non-sumerian documents

    Args:
        raw_documents (List[str]): A list of strings where each string is the entire raw dosument

    Returns:
        List[str]: A list of strings of only sumerian (i.e. non Akkadian) documents
    """

    filtered_docs = []

    for document in raw_documents:

        # Performs a regex to extract the language
        match = re.compile(r'#atf: lang\s+(\w+)\s').search(document)
        if match:

            # Only add the document if the language is 'sux' or sumerian
            if match.group(1) == 'sux':
                filtered_docs.append(document)

    return filtered_docs

def process_documents(documents):
        """Takes the raw document data in ATF format and converts it into a tokenized sequence of sumerian words

        Args:
            document (List[str]): A list of lines containing document information

        Returns:
            List[str]: A list of tokens that have been processed or the empty string if the document is invalid
        """

        processed_docs = []
        for document in documents:
            # This matches all lines of sumerian text within the document
            match = re.compile(r'[^\s#]+\.\s(.*)$', re.MULTILINE).findall(document)
            if match:

                # Joins the match back into a string, sanitizes it and appends it to the processed docs
                doc = ''.join(match)
                clean_doc = clean_text(doc)
                processed_docs.append(clean_doc)

        return processed_docs

def extract_translations(filtered_documents):
    """Extracts lines with an english translation 

    Args:
        filtered_documents (_type_): _description_

    Returns:
        _type_: _description_
    """
    sumerian_english_lines = []

    for document in filtered_documents:

        # This regular expression matches all instance of a sumerian line with an english translation underneath
        match = re.compile(r'[^\s#]+\.\s(.*)$\s#tr.en:(.*)$', re.MULTILINE).findall(document)
        if match:
            # Match returns a list of sumerian-english tuples
            for sumerian, english in match:
                
                
                # Sanitize the text
                sumerian_line = clean_text(sumerian).strip()

                # Deletes everything in the english line that is within a parentheses
                english = re.sub(r'\(.*\)', '' ,english)
                
                # If the english line is all whitespace, skip it
                if re.match(r'\s+\Z', english):
                    continue
                
                parallel_line = sumerian_line + '||' + english

                # Delete all lines with '...' since that represents unidentified symbols
                if '...' not in parallel_line and 'xxx' not in parallel_line:
                    sumerian_english_lines.append(parallel_line)

    # Deletes all the duplicates in the text
    sumerian_english_lines = set(sumerian_english_lines)

    # Split the list into a list for english lines and a list for the sumerian lines
    sum_lines = [sentence.split("||")[0] for sentence in sumerian_english_lines]
    eng_lines = [sentence.split("||")[1] for sentence in sumerian_english_lines]

    return sum_lines, eng_lines

def clean_text(text):
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
        # The "_" character indicates a logogram
        # None of these characters exist in the vocab so they can be removed
        text = re.sub(r'[#\[\]<>|?!_]', '', text)
        
        # This removes any occurence of "($ 'some text' $)" since these are generally inline comments such as
        # "blank space"
        text = re.sub(r'\(\$.*\$\)', '', text)
        
        # This removes any occurence of a stand-alone ellipses "..." at the beginning of the text or elsewhere
        # These ellipses indicate that an undeterminable number of signs may be missing from the tablet
        text = re.sub(r'(\s|^)(\.\.\.\s)+', ' ', text)

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

        # Removes any occurence of @ which generally is some comment about the sign
        text = re.sub('@\S', '', text)

        return text

def write_documents(documents, output_name):
    """Writes the documents to the data folder"""

    with open(f'data/{output_name}', 'w') as outfile:
        documents = [document + '\n' for document in documents]
        outfile.writelines(documents)

def split_dataset(path):
    """ Splits the dataset found at path into a train/dev/test set"""

    # Opens the dataset and extracts all the lines from it
    with open(path, 'r') as dataset:
        lines = [line.strip() for line in dataset]

    num_samples = len(lines)

    # Gets how many samples make up 10 percent of the data
    dev_test_split = num_samples // 10

    # Splits the lines into a train/dev/test split
    dev_set = lines[:dev_test_split]
    test_set = lines[dev_test_split:dev_test_split*2]
    train_set = lines[dev_test_split*2:]

    # Puts the datasets in a list and constructs the suffixes to differentiate the dataset paths
    datasets = [train_set, dev_set, test_set]
    path_suffixes = ['_train', '_dev', '_test']

    # Iterates through the datasts
    for dataset, suffix in zip(datasets, path_suffixes):

        # Constructs a new path that includes the train/dev/test suffix
        original_path = Path(path)
        new_path = f"data/{original_path.stem}{suffix}{original_path.suffix}"

        with open(new_path, 'w') as outfile:
            
            # Adds a newline to the end of every line and writes it to file
            lines = [line + '\n' for line in dataset]
            outfile.writelines(lines)


def main():
    raw_data_path = "data/raw_sumerian.atf"
    process_data(raw_data_path, 'clean_sumerian.txt')


if __name__ == "__main__":
    main()