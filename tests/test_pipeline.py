"""Fast unit tests that need no data or GPU:  python -m pytest -q tests/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medsupport.config import EXTRA_EVIDENCES, INDICATION_TO_EVIDENCE, THORACIC_LABELS  # noqa: E402
from medsupport.explain import GradCAM  # noqa: E402
from medsupport.models import ClinicalMLP, CrossModalFusionNet, SymptomEncoder, SymptomEncoderClassifier  # noqa: E402
from medsupport.triage import triage  # noqa: E402


def test_symptom_encoder_shapes_and_mask():
    enc = SymptomEncoder(n_symptoms=10, dim=16)
    x = torch.zeros(2, 10)
    x[0, 3] = 1
    tokens, mask, pooled = enc(x)
    assert tokens.shape == (2, 11, 16) and pooled.shape == (2, 16)
    assert mask[0, 0].item() is False and mask[0, 4].item() is False and mask[0, 5].item() is True
    assert mask[1, 1:].all()            # patient with no symptoms: all symptom tokens masked
    assert torch.isfinite(pooled).all()


def test_fusion_forward_and_attention():
    net = CrossModalFusionNet(n_symptoms=12, n_labels=5, dim=32, heads=4, pretrained=False, freeze_until=None).eval()
    img = torch.randn(2, 3, 224, 224)
    sym = torch.zeros(2, 12)
    sym[1, [0, 5]] = 1
    out = net(img, sym, return_attn=True)
    assert out["logits"].shape == (2, 5) and out["img_logits"].shape == (2, 5) and out["cli_logits"].shape == (2, 5)
    assert out["attn_c2i"].shape == (2, 13, 50) and out["attn_i2c"].shape == (2, 1, 13)
    a = out["attn_i2c"][1, 0]
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-4)
    assert a[2].item() == 0.0            # masked (absent) symptom receives no attention
    assert torch.isfinite(out["logits"]).all()


def test_gradcam_produces_normalised_map():
    net = CrossModalFusionNet(n_symptoms=4, n_labels=3, dim=16, heads=2, pretrained=False, freeze_until=None)
    cam = GradCAM(net, net.image.features)
    img, sym = torch.randn(1, 3, 224, 224), torch.zeros(1, 4)
    heat = cam(lambda: net(img, sym)["logits"], 1, (224, 224))
    cam.remove()
    assert heat.shape == (224, 224) and heat.min() >= 0 and heat.max() <= 1


def test_clinical_mlp_and_classifier():
    m = ClinicalMLP(7, 3)
    assert m(torch.randn(4, 7)).shape == (4, 3)
    c = SymptomEncoderClassifier(9, 5, dim=16)
    logits, emb = c(torch.zeros(3, 9), return_embedding=True)
    assert logits.shape == (3, 5) and emb.shape == (3, 16)


def test_indication_patterns_compile_and_reference_known_codes():
    import re
    for pattern, codes in INDICATION_TO_EVIDENCE:
        re.compile(pattern)
        for c in codes:
            assert c.startswith("E_") or c in EXTRA_EVIDENCES, c
    for p in THORACIC_LABELS.values():
        re.compile(p)


def test_triage_levels_monotone():
    low = triage(0.05, 0.05, 0.8, set(), {}, age=30)
    high = triage(0.9, 0.7, 0.0, {"breathlessness"}, {}, age=70)
    assert low.level == "LOW" and high.level == "EMERGENCY"
    assert low.score < high.score <= 100
    assert any("common cold" in r for r in low.reasons)
