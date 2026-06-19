"""Shared Transformer Reasoning Model (SPEC §2.3).

Both LVMM and the Baseline use this identical architecture; they differ only in whether
``visual_tokens`` are entity-injected (LVMM) or raw F_core (Baseline / LVMM-NoDB).

    visual: 196 F_core tokens -> Linear(768->256), prepend [CLS] -> 197 tokens
    question: learned embedding table (vocab 3000) -> 256, with positional embeddings
    concat [visual || question] -> 6x pre-norm Transformer encoder blocks
    [CLS] output -> Linear head -> logits  (cross-entropy loss)

~4.2M trainable parameters at the default config.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReasoningModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        n_answers: int = 28,
        q_vocab_size: int = 3000,
        max_q_len: int = 30,
        dropout: float = 0.1,
        n_visual_tokens: int = 196,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_visual_tokens = n_visual_tokens
        self.max_q_len = max_q_len
        self.pad_idx = pad_idx

        # Visual stream.
        self.visual_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.visual_pos = nn.Parameter(torch.zeros(1, n_visual_tokens + 1, d_model))

        # Question stream.
        self.q_embed = nn.Embedding(q_vocab_size, d_model, padding_idx=pad_idx)
        self.q_pos = nn.Parameter(torch.zeros(1, max_q_len, d_model))

        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # pre-norm (SPEC §2.3)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_answers)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        nn.init.trunc_normal_(self.q_pos, std=0.02)

    def encode_cls(self, visual_tokens: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        """Return the [CLS] token embedding [B, d] (used by the §5.4 probe analysis)."""
        b = visual_tokens.shape[0]

        v = self.visual_proj(visual_tokens)                            # [B, 196, d]
        cls = self.cls_token.expand(b, -1, -1)                         # [B, 1, d]
        v = torch.cat([cls, v], dim=1) + self.visual_pos               # [B, 197, d]

        q_len = question_tokens.shape[1]
        q = self.q_embed(question_tokens) + self.q_pos[:, :q_len]      # [B, Q, d]

        x = torch.cat([v, q], dim=1)                                   # [B, 197+Q, d]
        x = self.input_norm(x)

        # Key padding mask: visual tokens all valid; mask padded question tokens.
        vis_mask = torch.zeros(b, v.shape[1], dtype=torch.bool, device=x.device)
        q_mask = question_tokens == self.pad_idx                       # [B, Q]
        key_padding_mask = torch.cat([vis_mask, q_mask], dim=1)        # [B, 197+Q]

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.final_norm(x[:, 0])                                # [B, d]

    def forward(self, visual_tokens: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        """visual_tokens: [B, 196, 768]; question_tokens: [B, Q] int -> logits [B, n_answers]."""
        cls_out = self.encode_cls(visual_tokens, question_tokens)
        return self.head(cls_out)                                      # [B, n_answers]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
