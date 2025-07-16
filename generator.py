# generator.py
import torch
import torch.nn as nn
from config import config

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.generator_hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.generator_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config.generator_hidden_dim, config.generator_num_heads, config.generator_ffn_dim, config.dropout) 
            for _ in range(config.generator_num_layers)
        ])

        self.ln = nn.LayerNorm(config.generator_hidden_dim)
        self.head = nn.Linear(config.generator_hidden_dim, config.vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.head(x)
        return logits
