# discriminator.py
import torch
import torch.nn as nn
from config import config
from generator import TransformerBlock # Reuse the transformer block

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads, config.ffn_dim, config.dropout) 
            for _ in range(config.num_layers)
        ])

        self.ln = nn.LayerNorm(config.hidden_dim)
        # The head outputs a single logit per token for "real" vs "fake"
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.head(x).squeeze(-1) # Squeeze the last dimension
        return logits
