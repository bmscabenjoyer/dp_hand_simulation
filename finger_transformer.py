"""Small transformer encoder for finger token sequences."""

import torch
import torch.nn as nn

from finger_tokenize import (
    FINGER_VOCAB, LANE_VOCAB, DURATION_BINS,
    NUM_FINGERS, NUM_LANES,
)

MAX_SEQ_LEN = 256


class FingerTransformer(nn.Module):
    """
    Transformer encoder over finger token sequences.

    Input : (B, T, 3) int64  — [finger_id, lane, dur_bin] per token
    Output: finger logits (B, T, NUM_FINGERS), lane logits (B, T, NUM_LANES)

    Call encode() to get a mean-pooled (B, d_model) embedding for similarity search.
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_layers: int = 4,
        n_heads:  int = 4,
        d_ff:     int = 512,
        dropout:  float = 0.1,
    ):
        super().__init__()

        # Per-field embeddings
        self.finger_emb = nn.Embedding(FINGER_VOCAB, 32, padding_idx=FINGER_VOCAB - 1)
        self.lane_emb   = nn.Embedding(LANE_VOCAB,   32, padding_idx=LANE_VOCAB - 1)
        self.dur_emb    = nn.Embedding(DURATION_BINS, 32)

        self.input_proj = nn.Linear(96, d_model)
        self.pos_emb    = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.finger_head = nn.Linear(d_model, NUM_FINGERS)
        self.lane_head   = nn.Linear(d_model, NUM_LANES)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shared embedding + positional encoding. input_ids: (B, T, 3)"""
        B, T, _ = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = torch.cat([
            self.finger_emb(input_ids[:, :, 0]),
            self.lane_emb  (input_ids[:, :, 1]),
            self.dur_emb   (input_ids[:, :, 2]),
        ], dim=-1)                         # (B, T, 96)
        return self.input_proj(x) + self.pos_emb(pos)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        finger_logits : (B, T, NUM_FINGERS)
        lane_logits   : (B, T, NUM_LANES)
        """
        x = self._embed(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        return self.finger_head(x), self.lane_head(x)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pooled sequence embedding for similarity search. Returns (B, d_model)."""
        x = self._embed(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)


def model_size(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.2f}M params"
