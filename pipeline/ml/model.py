"""Model definition: a small decoder-only (causal) Transformer over the pitch
token vocabulary defined in data_prep.py.

Chosen over an LSTM for this project because the training data spans three
very different sources (piano theory, general violin-relevant material,
solo violin etudes) combined via curriculum learning — attention lets the
model use long-range context (phrase repetition, motif return) that a
melodic line depends on, and the same architecture/vocab carries cleanly
across all three curriculum stages without any structural change between
stages, only which data it's fed and which layers are frozen/unfrozen.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pipeline.ml.data_prep import PAD, VOCAB_SIZE


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.), added to the
    token embeddings before the Transformer stack. Fixed, not learned — with
    the relatively small/mixed dataset here, learned positional embeddings
    have more parameters to overfit for no expected benefit at this scale."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


def generate_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Bool mask: True where attention is forbidden (future positions),
    False where it's allowed. Position i may attend to positions <= i only.
    Bool (not float/additive) so it matches the dtype of the padding mask —
    PyTorch's MultiheadAttention deprecates mixing a float and a bool mask
    in the same call."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class NoteTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        pad_token: int = PAD,
    ):
        super().__init__()
        self.pad_token = pad_token
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN: more stable training at this depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm_out = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        # Tie input/output embeddings — halves the embedding-related
        # parameter count and is standard practice for this size of model.
        self.output_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [batch, seq_len] (LongTensor)
        Returns logits: [batch, seq_len, vocab_size]"""
        seq_len = input_ids.size(1)
        device = input_ids.device

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        causal_mask = generate_causal_mask(seq_len, device=device)
        padding_mask = input_ids == self.pad_token  # [batch, seq_len], True = ignore

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm_out(x)
        return self.output_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive sampling for a single sequence. prompt_ids: [seq_len]
        (no batch dim). Returns the prompt extended by up to max_new_tokens."""
        from pipeline.ml.data_prep import EOS  # local import avoids a hard dependency at module load

        self.eval()
        device = prompt_ids.device
        generated = prompt_ids.clone().unsqueeze(0)  # [1, seq_len]

        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == EOS:
                break

        return generated.squeeze(0)
