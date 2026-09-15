"""SASRec (Self-Attentive Sequential Recommendation) на PyTorch."""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        emb_dim: int,
        max_len: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        items = self.item_embedding(input_ids)
        pos_embs = self.pos_embedding(positions)
        x = items + pos_embs
        x = self.layer_norm(x)

        seq_len = input_ids.size(1)
        # Causal mask: True = заблокировано (нельзя смотреть вперёд).
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        padding_mask = input_ids.eq(0)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Для right padding берём последнюю реальную позицию каждой последовательности.
        lengths = (~padding_mask).sum(dim=1)
        last_indices = lengths - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        user_vecs = encoded[batch_indices, last_indices]
        return user_vecs