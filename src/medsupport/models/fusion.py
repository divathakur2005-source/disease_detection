"""Intermediate cross-modal attention fusion.

Two directions of multi-head cross-attention:
  * clinical -> image : symptom tokens (queries) attend over the 7x7 image
                        patch tokens, so each reported symptom "looks" at the
                        lung regions that could explain it.
  * image -> clinical : image summary token attends over the symptom tokens, so
                        the visual evidence is re-weighted by what the patient
                        reports (e.g. fever + cough makes an opacity more
                        likely to be read as infection).
The two attended summaries are harmonised (LayerNorm + MLP) and concatenated
with the unimodal pooled vectors before the multi-label head.

Returns attention weights so the app can show *which* symptoms drove the
decision and which image regions each symptom attended to.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .clinical import SymptomEncoder
from .imaging import ImageEncoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, q, kv, kv_mask=None):
        a, w = self.attn(self.norm1(q), kv, kv, key_padding_mask=kv_mask, need_weights=True,
                         average_attn_weights=True)
        q = q + a
        q = q + self.ff(self.norm2(q))
        return q, w


class CrossModalFusionNet(nn.Module):
    def __init__(self, n_symptoms: int, n_labels: int, dim: int = 256, heads: int = 4, dropout: float = 0.2,
                 pretrained: bool = True, freeze_until: str | None = "denseblock3"):
        super().__init__()
        self.image = ImageEncoder(n_classes=None, pretrained=pretrained, dim=dim, freeze_until=freeze_until)
        self.clinical = SymptomEncoder(n_symptoms, dim=dim)
        self.img_cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.img_cls, std=0.02)
        self.img_pos = nn.Parameter(torch.zeros(1, 50, dim))     # 1 + 7*7 tokens at 224 px
        nn.init.trunc_normal_(self.img_pos, std=0.02)

        self.c2i = CrossAttentionBlock(dim, heads, dropout)      # clinical queries -> image keys
        self.i2c = CrossAttentionBlock(dim, heads, dropout)      # image query     -> clinical keys

        self.harmonise = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout))
        fused_in = dim + self.image.feat_channels + dim          # harmonised + image pooled + clinical pooled
        self.head = nn.Sequential(nn.Linear(fused_in, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(256, n_labels))
        # auxiliary unimodal heads (help balance modality learning; also give baselines for free)
        self.img_head = nn.Linear(self.image.feat_channels, n_labels)
        self.cli_head = nn.Linear(dim, n_labels)

    def forward(self, img: torch.Tensor, sym: torch.Tensor, return_attn: bool = False):
        img_tok, img_pooled, fmap = self.image.tokens(img)                       # (B,49,D), (B,1024)
        B = img.shape[0]
        img_tok = torch.cat([self.img_cls.expand(B, -1, -1), img_tok], dim=1)     # (B,50,D)
        img_tok = img_tok + self.img_pos[:, : img_tok.shape[1]]

        cli_tok, cli_mask, cli_pooled = self.clinical(sym)                        # (B,1+S,D)

        c_att, w_c2i = self.c2i(cli_tok, img_tok)                                 # symptoms look at image
        i_att, w_i2c = self.i2c(img_tok[:, :1], cli_tok, kv_mask=cli_mask)        # image looks at symptoms

        c_summary = c_att[:, 0]
        i_summary = i_att[:, 0]
        fused = self.harmonise(torch.cat([c_summary, i_summary], dim=-1))
        logits = self.head(torch.cat([fused, img_pooled, cli_pooled], dim=-1))
        out = {
            "logits": logits,
            "img_logits": self.img_head(img_pooled),
            "cli_logits": self.cli_head(cli_pooled),
        }
        if return_attn:
            out["attn_c2i"] = w_c2i          # (B, 1+S, 50)  which image patch each symptom attends to
            out["attn_i2c"] = w_i2c          # (B, 1, 1+S)   which symptoms the image summary attends to
            out["fmap"] = fmap
        return out
