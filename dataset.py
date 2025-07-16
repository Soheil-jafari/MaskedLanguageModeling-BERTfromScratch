# dataset.py
import torch
from torch.utils.data import Dataset
import random
from config import config

class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.data = [tokenizer.encode(line).ids for line in texts if line.strip()]
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.mask_token_id = tokenizer.token_to_id("[MASK]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The tokenization should already be done
        input_ids = self.data[idx]
        
        # Truncate to max_seq_len
        input_ids = input_ids[:config.max_seq_len]

        labels = torch.tensor(input_ids, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create the mask for the generator
        rand = torch.rand(input_ids.shape)
        # We only mask non-special tokens
        mask = (rand < config.mask_prob) & (input_ids != self.tokenizer.token_to_id("[CLS]")) & \
               (input_ids != self.tokenizer.token_to_id("[SEP]")) & (input_ids != self.pad_token_id)
        
        # Where mask is True, set the label to be the original token, otherwise -100
        generator_labels = torch.where(mask, labels, -100)
        
        input_ids[mask] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "labels": generator_labels,
        }
