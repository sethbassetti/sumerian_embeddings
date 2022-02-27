import os
import time
from sumerian_data import SumerianDataset
from torch.utils.data import DataLoader

def main():

    # Constructs the path to the sumerian sentences corpusl
    data_path = os.path.join("data/sumerian_document_set.atf")
    vocab_path = "data/sumerian_vocab_list"

    # Constructs the dataset and dataloader
    train_set = SumerianDataset(data_path, vocab_path)

    # Custom collate function prevents the automatic attempt to tuple dataset into target/label pairs
    train_loader = DataLoader(train_set, collate_fn=lambda x: x[0])

    # Very simple example just to see how data is loaded
    i = 0
    # Iterates through the first 50 example sentences and prints them out
    start = time.time()
    for example in train_loader:
        for token in example:
            if not train_set.in_vocab(token):
                print(token)
                print(example)
                print(i)
                exit()
        i += 1

    print(str(time.time() - start))
    




if __name__ == "__main__":
    main()