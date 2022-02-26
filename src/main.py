import os
from sumerian_data import SumerianDataset
from torch.utils.data import DataLoader

def main():

    # Constructs the path to the sumerian sentences corpusl
    data_path = os.path.join("data/sumerian_sentences.txt")

    # Constructs the dataset and dataloader
    train_set = SumerianDataset(data_path)
    train_loader = DataLoader(train_set)

    # Very simple example just to see how data is loaded
    i = 0
    
    # Iterates through the first 50 example sentences and prints them out
    for example in train_loader:
        print(example)
        i += 1
        if i > 50:
            break





if __name__ == "__main__":
    main()