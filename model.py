# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim)
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.ln = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

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
