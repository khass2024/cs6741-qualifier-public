import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)
        print(f"Loaded {len(self.data)} samples for {split}.")

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        input_file = os.path.join(data_folder, f"{split}.nl")
        target_file = os.path.join(data_folder, f"{split}.sql") if split != "test" else None
        
        data = []
        if os.path.exists(input_file):
            with open(input_file, 'r') as infile:
                if split == "test":
                    # For the test set, only process input
                    for line in infile:
                        input_tokens = tokenizer.encode(line.strip(), add_special_tokens=True)
                        data.append((input_tokens, None))
                else:
                    # For train/dev, process both input and target
                    if os.path.exists(target_file):
                        with open(target_file, 'r') as targetfile:
                            for line, target in zip(infile, targetfile):
                                input_tokens = tokenizer.encode(line.strip(), add_special_tokens=True)
                                target_tokens = tokenizer.encode(target.strip(), add_special_tokens=True)
                                data.append((input_tokens, target_tokens))
                    else:
                        print(f"Target file {target_file} not found.")
        else:
            print(f"Input file {input_file} not found.")
        
        return data
    
    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_mask, decoder_inputs, decoder_targets = [], [], [], []
    
    for input_ids, target_ids in batch:
        encoder_ids.append(torch.tensor(input_ids))
        encoder_mask.append(torch.tensor([1] * len(input_ids)))  # Mask out non-padding tokens
        
        decoder_inputs.append(torch.tensor([0] + target_ids[:-1]))  # Shift the target tokens to the right by 1 for decoder input
        decoder_targets.append(torch.tensor(target_ids))

    # Pad sequences for dynamic batching
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

    initial_decoder_inputs = decoder_inputs[:, 0]  # First token for evaluation

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_mask = [], []

    for input_ids, _ in batch:
        encoder_ids.append(torch.tensor(input_ids))
        encoder_mask.append(torch.tensor([1] * len(input_ids)))

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)

    initial_decoder_inputs = torch.zeros(encoder_ids.size(0), dtype=torch.long)  # First token for T5

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x