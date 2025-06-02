# dataset.py
import torch
from torch.utils.data import Dataset
import random

class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = [tokenizer.encode(line) for line in texts if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        input_ids = input_ids[:self.config.max_seq_len]

        labels = input_ids.copy()
        mask_token_id = self.tokenizer.vocab["[MASK]"]

        for i in range(1, len(input_ids) - 1):  # exclude [CLS] and [SEP]
            if random.random() < self.config.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = mask_token_id
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, self.config.vocab_size - 1)
                # else keep original
            else:
                labels[i] = -100  # Ignore token in loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
