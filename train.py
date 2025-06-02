# train.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
from tqdm import tqdm
from config import config
from model import MaskedLanguageModel
from tokenizer import Tokenizer
from dataset import MaskedTextDataset


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=config.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_data():
    with open(config.train_text_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer()
    tokenizer.build_vocab(config.train_text_path)

    train_texts = load_data()
    dataset = MaskedTextDataset(train_texts, tokenizer, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    model = MaskedLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        total_loss = 0

        for input_ids, labels in loop:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

        save_path = os.path.join(config.model_save_path, f"model_epoch{epoch + 1}.pt")
        os.makedirs(config.model_save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == '__main__':
    train()
