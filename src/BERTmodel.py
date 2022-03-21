import torch
from transformers import BertTokenizer, BertModel, BertConfig, AdamW, BertForMaskedLM
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from sumerian_data import SumerianDataset
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import os

DEVICE = torch.device("cuda")

DATA_PATH = r"../data/sumerian_document_set.atf"
VOCAB_PATH = r"../data/sumerian_vocab_list"

def save_corpus(save_path: str, dataset: torch.utils.data.Dataset):
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print("Directory path already exists.")

    docs = dataset.get_corpus()
    processed_doc = [" ".join(doc) for doc in docs]
    with open(os.path.join(save_path, r"processed_corpus.txt"), "w") as fp:
        fp.write("\n".join(processed_doc))


class SumarianBERTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: ByteLevelBPETokenizer, data_path: str, evaluate: bool = False):
        self.evaluate = evaluate
        self.tokenizer = tokenizer
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=512)
        self.training_labels, self.test_labels, self.training_mask, self.test_mask = self.__get_split(data_path)
    
    def __get_split(self, data_path: str) -> Tuple[list, list]:

        with open(data_path, "r") as file:
            lines = file.read().split("\n")

        lines_tokens = [line for line in self.tokenizer.encode_batch(lines)]

        mask = [x.attention_mask for x in lines_tokens]
        labels = [line.ids for line in lines_tokens]

        indices = np.random.permutation(len(labels))
        split = int(len(labels) * 0.8)
        training_idxs, test_idxs = indices[:split], indices[split:]

        training_labels, test_labels = [], []
        training_mask, test_mask = [], []

        for train_idx in training_idxs:
            training_labels.append(labels[train_idx])
            training_mask.append(mask[train_idx])

        for test_idx in test_idxs:
            test_labels.append(labels[test_idx])
            test_mask.append(mask[test_idx])

        return training_labels, test_labels, training_mask, test_mask

    def __len__(self):
        if self.evaluate:
            return len(self.test_labels)
        else:
            return len(self.training_labels)

    def __getitem__(self, i):
        if self.evaluate:
            return (
                torch.tensor(self.test_labels[i]).type(torch.float), 
                torch.tensor(self.test_mask[i]).type(torch.float)
                )
        else:
            return (
                torch.tensor(self.training_labels[i]).type(torch.float), 
                torch.tensor(self.training_mask[i]).type(torch.float)
                )


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    labels = [item[0] for item in batch]
    att_mask = [item[1] for item in batch]

    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in labels ]).to(DEVICE)

    ## padd
    labels = [ torch.Tensor(t).to(DEVICE) for t in labels ]
    labels = torch.nn.utils.rnn.pad_sequence(labels)

    att_mask = [ torch.Tensor(t).to(DEVICE) for t in att_mask ]
    att_mask = torch.nn.utils.rnn.pad_sequence(att_mask)

    ## compute mask
    mask = (labels != 0).to(DEVICE)

    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)

    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        input_ids[i, selection] = 3

    return labels.T, att_mask.T, input_ids.T, lengths.T, mask.T

# model = BertModel.from_pretrained("bert-base-multilingual-cased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

def main():
    SAVE_CORPUS=False
    MAKE_TOKENIZER=False

    dataset = SumerianDataset(DATA_PATH, VOCAB_PATH)

    save_path=r"../data/processed_data/"

    if SAVE_CORPUS:
        save_corpus(save_path, dataset)

    if MAKE_TOKENIZER:
        vocab_size = dataset.get_vocab_size()

        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=os.path.join(save_path, r"processed_corpus.txt"), 
            vocab_size=vocab_size, 
            min_frequency=2, 
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ])

        try:
            os.mkdir(r"../tokenizer/")
        except FileExistsError:
            print("Tokenizer directory path already exists.")

        tokenizer.save_model(r"../tokenizer/")

    tokenizer = ByteLevelBPETokenizer(
        "../tokenizer/vocab.json",
        "../tokenizer/merges.txt",
    )

    BERT_dataset = SumarianBERTDataset(
        tokenizer, 
        os.path.join(save_path, r"processed_corpus.txt"), 
        evaluate=False)

    
    BERT_train_loader = torch.utils.data.DataLoader(
        BERT_dataset, 
        batch_size=16, 
        shuffle=True,
        collate_fn=collate_fn_padd)

    config = BertConfig(
        vocab_size=dataset.get_vocab_size(),
        max_position_embeddings=512,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=2
    )

    # config = BertConfig(
    #     vocab_size=dataset.get_vocab_size(),
    #     max_position_embeddings=512,
    #     hidden_size=768,
    #     num_attention_heads=4,
    #     num_hidden_layers=4,
    #     type_vocab_size=1
    # )

    model = BertForMaskedLM(config)

    multling_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    multling_params = multling_model.state_dict()

    # Remove params that are a mismatch with current model.
    del multling_params["embeddings.position_ids"]
    del multling_params['embeddings.word_embeddings.weight'] 
    del multling_params['embeddings.position_embeddings.weight'] 
    del multling_params['embeddings.token_type_embeddings.weight'] 
    del multling_params['embeddings.LayerNorm.weight']
    del multling_params['embeddings.LayerNorm.bias']

    model.load_state_dict(multling_params, strict=False)

    model.to(DEVICE)

    print("Number of parameters: ", end="")
    print(model.num_parameters())

    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)

    epochs = 2

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(BERT_train_loader, leave=True)
        for batch in loop:
            optim.zero_grad()

            labels, attention_mask, input_ids, lengths, mask = batch

            input_ids.to(DEVICE)
            attention_mask.to(DEVICE)
            labels.to(DEVICE)
            
            outputs = model(input_ids.long(), attention_mask=attention_mask, labels=labels.long().contiguous())
            
            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


if __name__=="__main__":
    main()