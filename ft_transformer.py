"""
ft_transformer.py
==================
A small, dependency-light FT-Transformer (Feature Tokenizer Transformer)
implementation in plain PyTorch. Every model folder (placement, cgpa,
scholarship, mentoring, internship) imports this same class and just
points it at different columns / targets.

Reference idea: Gorishniy et al., "Revisiting Deep Learning Models for
Tabular Data" (FT-Transformer). This is a compact re-implementation,
not the pytorch-tabular package, so it trains fast on CPU.
"""

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Turns every raw column (numeric or categorical) into a d-dim token."""

    def __init__(self, n_continuous: int, categorical_cardinalities: list[int], d_token: int):
        super().__init__()
        self.n_continuous = n_continuous
        self.d_token = d_token

        # One learnable weight+bias per continuous feature (like a per-feature linear layer)
        if n_continuous > 0:
            self.cont_weight = nn.Parameter(torch.randn(n_continuous, d_token) * 0.02)
            self.cont_bias = nn.Parameter(torch.zeros(n_continuous, d_token))

        # One embedding table per categorical feature
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in categorical_cardinalities]
        )

        # CLS token, prepended to the sequence of feature tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_cont.shape[0] if self.n_continuous > 0 else x_cat.shape[0]
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if self.n_continuous > 0:
            # x_cont: (B, n_continuous) -> (B, n_continuous, d_token)
            cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight + self.cont_bias
            tokens.append(cont_tokens)

        for i, emb in enumerate(self.cat_embeddings):
            cat_tok = emb(x_cat[:, i]).unsqueeze(1)  # (B, 1, d_token)
            tokens.append(cat_tok)

        return torch.cat(tokens, dim=1)  # (B, 1 + n_features, d_token)


class FTTransformer(nn.Module):
    """Feature Tokenizer + standard Transformer encoder + prediction head."""

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: list[int],
        d_token: int = 32,
        n_heads: int = 4,
        n_blocks: int = 3,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_continuous, categorical_cardinalities, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_outputs),
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_cont, x_cat)
        encoded = self.encoder(tokens)
        cls_output = encoded[:, 0, :]  # use the CLS token's representation
        return self.head(cls_output).squeeze(-1)
