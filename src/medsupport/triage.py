"""Rule-based triage engine.

Takes the probabilities produced by the three predictors and returns a
priority level, the reasons that fired, and an advisory. The rules encode
the comorbidity interactions the project is about (e.g. diabetes raises the
severity of a suspected chest infection), and they are deliberately simple and
transparent so a clinician can audit them.

This is decision *support*. It is not a diagnosis and must not be used without
qualified medical review.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TriageResult:
    level: str                       # EMERGENCY / HIGH / MODERATE / LOW
    score: int                       # 0-100, higher = more urgent
    reasons: list[str] = field(default_factory=list)
    advisory: list[str] = field(default_factory=list)
    referral: str = ""


LEVELS = ["LOW", "MODERATE", "HIGH", "EMERGENCY"]


def triage(
    p_chest_infection: float,
    p_diabetes: float,
    p_common_cold: float,
    symptoms: set[str] | None = None,
    thoracic_probs: dict[str, float] | None = None,
    age: float | None = None,
) -> TriageResult:
    symptoms = {s.lower() for s in (symptoms or set())}
    thoracic_probs = thoracic_probs or {}
    score = 0
    reasons: list[str] = []
    advisory: list[str] = []

    # --- red-flag symptoms ------------------------------------------------
    red_flags = {"breathlessness", "chest_pain", "hemoptysis", "hypoxia", "syncope"}
    hit = red_flags & symptoms
    if hit:
        score += 25
        reasons.append(f"Red-flag symptoms reported: {', '.join(sorted(hit))}")

    # --- chest infection -----------------------------------------------------
    if p_chest_infection >= 0.7:
        score += 40
        reasons.append(f"High probability of lower-respiratory infection on X-ray ({p_chest_infection:.0%})")
    elif p_chest_infection >= 0.4:
        score += 20
        reasons.append(f"Possible lower-respiratory infection on X-ray ({p_chest_infection:.0%})")

    # --- metabolic comorbidity multiplier -------------------------------------
    if p_diabetes >= 0.5:
        reasons.append(f"Elevated diabetes risk ({p_diabetes:.0%})")
        advisory.append("Confirm HbA1c / fasting glucose; diabetes impairs immune response and worsens pneumonia outcomes.")
        if p_chest_infection >= 0.4:
            score += 20
            reasons.append("Comorbidity interaction: suspected chest infection in a patient with elevated diabetes risk")
        else:
            score += 10

    # --- other thoracic findings -------------------------------------------
    for name, p in thoracic_probs.items():
        if name in ("pleural_effusion", "cardiomegaly") and p >= 0.5:
            score += 10
            reasons.append(f"{name.replace('_', ' ').title()} suspected ({p:.0%}) - cardiology/pulmonology review")

    # --- age ---------------------------------------------------------------
    if age is not None and age >= 65:
        score += 10
        reasons.append("Age >= 65")

    # --- benign path -------------------------------------------------------
    if p_common_cold >= 0.5 and p_chest_infection < 0.4 and not hit:
        reasons.append(f"Symptom profile consistent with common cold ({p_common_cold:.0%}); chest X-ray not suggestive of infection")
        advisory.append("Rest, fluids, antipyretics; return if fever persists > 3 days or breathing becomes difficult.")

    score = min(score, 100)
    if score >= 70:
        level, referral = "EMERGENCY", "Immediate physician review; consider emergency department"
    elif score >= 45:
        level, referral = "HIGH", "Same-day specialist referral (pulmonology / internal medicine)"
    elif score >= 20:
        level, referral = "MODERATE", "Outpatient consultation within 48 hours"
    else:
        level, referral = "LOW", "Self-care with safety-net advice"
    if not reasons:
        reasons.append("No concerning findings from symptoms, vitals or imaging")
    return TriageResult(level=level, score=score, reasons=reasons, advisory=advisory, referral=referral)
