import os
from sumerian_data import SumerianDataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec


def main():

    # Constructs the path to the sumerian sentences corpusl
    data_path = os.path.join("data/clean_sumerian.txt")
    vocab_path = "data/sumerian_vocab_list"

    # Constructs the dataset and dataloader
    train_set = SumerianDataset(data_path, vocab_path)

    # Custom collate function prevents the automatic attempt to tuple dataset into target/label pairs
    train_loader = DataLoader(train_set, collate_fn=lambda x: x[0])

    sentences = [sentence for sentence in train_loader]
    #print(sentences[:50])
    model = Word2Vec(sentences=sentences, vector_size=100, window=10, min_count=3, workers=4)
    #model.save("word2vec.model")
    print(model.wv.most_similar('udu'))

if __name__ == "__main__":
    main()