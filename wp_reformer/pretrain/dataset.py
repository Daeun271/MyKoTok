import warnings
warnings.filterwarnings("ignore")

import os
import re
import kss
import json
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

class DatasetForMLM(Dataset):
    def __init__(self, tokenizer, max_len, path):
        logging.info('start wiki data load')

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = []
        
        if os.path.isdir(path):
            file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
        elif os.path.isfile(path):
            file_list = [path]
        else:
            raise ValueError('path is not valid')

        if len(file_list) > 1:
            file_iterable = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        else:
            file_iterable = file_list

        for file_path in file_iterable:
            with open(file_path, 'r', encoding='utf-8') as data_file:
                for line in tqdm(data_file,
                                 desc='Data load for pretraining',
                                 position=1, leave=True):
                    line = line.rstrip()
                    self.docs.append(line)

        logging.info('complete data load')
        
    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone() # inputs will be modified to mask tokens while keeping labels intact to track which tokens are masked.
        
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ] # This returns a list where special tokens (e.g., [CLS], [SEP], [PAD]) are marked as 1 and the rest as 0.
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0) # converts special_tokens_mask to a PyTorch tensor and fills the probability_matrix with 0.0 where special tokens are present.
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 # For all unmasked tokens (~masked_indices), labels is set to -100. This ensures that only masked tokens contribute to the loss during training.

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        if pad:
            input_pads = self.max_len - inputs.shape[-1]
            label_pads = self.max_len - labels.shape[-1]

            inputs = F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads), value=self.tokenizer.pad_token_id)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        '''tokenizes a list of input IDs and converts them into a PyTorch tensor'''
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=False, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        return len(self.docs)
    def __getitem__(self, idx):
        '''tokenizes the document of the given index, applies masking, and returns necessary tensors.'''
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs,pad=True)

        inputs= inputs.squeeze()
        inputs_mask = inputs != 0 # creates a mask tensor where 1 represents a real token and 0 represents a padding token. This is used to ignore padding tokens during training.
        labels= labels.squeeze()

        return inputs, inputs_mask, labels
