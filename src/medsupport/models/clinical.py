"""Clinical (tabular) branch.

* ClinicalMLP   - generic MLP classifier used for both the symptom->disease model
                  (binary symptom vector -> 41 diseases) and the diabetes-risk
                  model (8 vitals -> diabetic / non-diabetic).
* SymptomEncoder- turns a binary symptom vector into (a) a pooled 256-d
                  embedding and (b) a *set of symptom tokens* (one learned
                  embedding per present symptom) that the fusion model attends over.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)
        self.embed_dim = d

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.backbone(x)
        logits = self.head(z)
        return (logits, z) if return_embedding else logits


class SymptomEncoder(nn.Module):
    """Binary symptom vector -> token set + pooled embedding.

    Each symptom index has a learned embedding. For a patient, the tokens are the
    embeddings of the symptoms that are present (masked otherwise), plus a
    learned [CLS]-style summary token that always exists so an empty symptom
    set still produces a valid representation.
    """

    def __init__(self, n_symptoms: int, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_symptoms = n_symptoms
        self.dim = dim
        self.symptom_emb = nn.Embedding(n_symptoms, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mlp = nn.Sequential(nn.Linear(n_symptoms, dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor):
        """x: (B, n_symptoms) float {0,1}.

        Returns tokens (B, 1+n_symptoms, dim), key_padding_mask (B, 1+n_symptoms) with
        True = ignore, and pooled (B, dim).
        """
        B = x.shape[0]
        idx = torch.arange(self.n_symptoms, device=x.device)
        present = x > 0.5
        present_f = present.float().unsqueeze(-1)                             # (B, S, 1)
        tok = self.symptom_emb(idx).unsqueeze(0).expand(B, -1, -1) * present_f  # (B, S, D), absent -> 0
        # summary token = learned [CLS] + MLP over the whole vector + mean of present symptom embeddings,
        # so the embedding table receives gradients during unimodal pre-training as well.
        mean_tok = tok.sum(1) / present_f.sum(1).clamp(min=1.0)
        cls = (self.cls.squeeze(1) + self.mlp(x) + mean_tok).unsqueeze(1)     # (B, 1, D)
        tokens = self.norm(torch.cat([cls, tok], dim=1))
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~present], dim=1)
        pooled = tokens[:, 0]
        return tokens, mask, pooled


class SymptomEncoderClassifier(nn.Module):
    """SymptomEncoder + linear head. Used to pre-train the encoder on DDXPlus
    (49-way diagnosis) before it is transferred into the fusion network, and
    as the clinical-only baseline on Open-I (multi-label thoracic findings)."""

    def __init__(self, n_symptoms: int, n_classes: int, dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = SymptomEncoder(n_symptoms, dim=dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.embed_dim = dim

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        _, _, pooled = self.encoder(x)
        logits = self.head(pooled)
        return (logits, pooled) if return_embedding else logits
