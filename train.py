# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import os
from tqdm import tqdm
from config import config
from tokenizer import get_tokenizer
from dataset import MaskedTextDataset
from generator import Generator
from discriminator import Discriminator

def get_scheduler(optimizer, num_training_steps):
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - config.warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=config.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {"input_ids": input_ids, "labels": labels}

def train():
    device = torch.device(config.device)
    tokenizer = get_tokenizer()
    config.pad_token_id = tokenizer.token_to_id("[PAD]")

    print("Loading data...")
    with open(config.dataset_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    dataset = MaskedTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Use a single optimizer for both models
    optimizer = torch.optim.AdamW(list(generator.parameters()) + list(discriminator.parameters()), lr=config.learning_rate)
    num_training_steps = len(dataloader) * config.num_epochs
    scheduler = get_scheduler(optimizer, num_training_steps)

    print("Starting training...")
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        total_gen_loss = 0
        total_disc_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            generator_labels = batch["labels"].to(device)
            
            # Generator forward pass
            generator_logits = generator(input_ids)
            
            # Generator loss
            gen_loss = F.cross_entropy(generator_logits.view(-1, config.vocab_size), generator_labels.view(-1), ignore_index=-100)
            
            # Generate fake tokens to be used by discriminator
            with torch.no_grad():
                # Get the probabilities from the generator's output
                pred_probs = F.softmax(generator_logits, dim=-1)
                # Sample from the distribution
                sampled_tokens = torch.multinomial(pred_probs.view(-1, config.vocab_size), 1).view(input_ids.shape)
            
            # Create the discriminator's input and labels
            # We only replace the tokens that were originally masked
            masked_positions = (generator_labels != -100)
            discriminator_input = torch.where(masked_positions, sampled_tokens, input_ids)
            discriminator_labels = masked_positions.float() # 1 for replaced, 0 for original

            # Discriminator forward pass
            discriminator_logits = discriminator(discriminator_input)
            
            # Discriminator loss
            disc_loss = F.binary_cross_entropy_with_logits(discriminator_logits, discriminator_labels)
            
            # Combined loss
            total_loss = config.generator_weight * gen_loss + config.discriminator_weight * disc_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(generator.parameters()) + list(discriminator.parameters()), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()

        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_loss = total_disc_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Avg Gen Loss: {avg_gen_loss:.4f} | Avg Disc Loss: {avg_disc_loss:.4f}")

        # Save models
        os.makedirs(config.model_save_path, exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(config.model_save_path, f"generator_epoch_{epoch+1}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(config.model_save_path, f"discriminator_epoch_{epoch+1}.pt"))
        print(f"Models saved for epoch {epoch+1}")

if __name__ == '__main__':
    train()
