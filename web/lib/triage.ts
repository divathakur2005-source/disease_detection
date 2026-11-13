/**
 * Rule-based triage engine - a direct port of src/medsupport/triage.py so the
 * demo and the Python system give identical recommendations.
 *
 * The rules encode the comorbidity interaction the project is about: a
 * suspected chest infection in a patient with elevated diabetes risk is
 * escalated above either finding alone, because diabetes impairs immune
 * response and worsens pneumonia outcomes.
 */

export type TriageLevel = "LOW" | "MODERATE" | "HIGH" | "EMERGENCY";

export type TriageResult = {
  level: TriageLevel;
  score: number;
  reasons: string[];
  advisory: string[];
  referral: string;
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function triage(opts: {
  chestInfection: number;
  diabetes: number;
  commonCold: number;
  symptoms: Set<string>;
  thoracic?: Record<string, number>;
  age?: number;
}): TriageResult {
  const { chestInfection, diabetes, commonCold, symptoms, thoracic = {}, age } = opts;
  let score = 0;
  const reasons: string[] = [];
  const advisory: string[] = [];

  const redFlags = ["breathlessness", "chest_pain", "hemoptysis", "hypoxia", "syncope"];
  const hit = redFlags.filter((f) => symptoms.has(f));
  if (hit.length) {
    score += 25;
    reasons.push(`Red-flag symptoms reported: ${hit.join(", ")}`);
  }

  if (chestInfection >= 0.7) {
    score += 40;
    reasons.push(`High probability of lower-respiratory infection (${pct(chestInfection)})`);
  } else if (chestInfection >= 0.4) {
    score += 20;
    reasons.push(`Possible lower-respiratory infection (${pct(chestInfection)})`);
  }

  if (diabetes >= 0.5) {
    reasons.push(`Elevated diabetes risk (${pct(diabetes)})`);
    advisory.push(
      "Confirm HbA1c / fasting glucose; diabetes impairs immune response and worsens pneumonia outcomes."
    );
    if (chestInfection >= 0.4) {
      score += 20;
      reasons.push("Comorbidity interaction: suspected chest infection with elevated diabetes risk");
    } else {
      score += 10;
    }
  }

  for (const [name, p] of Object.entries(thoracic)) {
    if ((name === "pleural_effusion" || name === "cardiomegaly") && p >= 0.5) {
      score += 10;
      const pretty = name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      reasons.push(`${pretty} suspected (${pct(p)}) - cardiology/pulmonology review`);
    }
  }

  if (age !== undefined && age >= 65) {
    score += 10;
    reasons.push("Age >= 65");
  }

  if (commonCold >= 0.5 && chestInfection < 0.4 && !hit.length) {
    reasons.push(
      `Symptom profile consistent with common cold (${pct(commonCold)}); no sign of lower-respiratory infection`
    );
    advisory.push("Rest, fluids, antipyretics; return if fever persists > 3 days or breathing becomes difficult.");
  }

  score = Math.min(score, 100);
  let level: TriageLevel;
  let referral: string;
  if (score >= 70) {
    level = "EMERGENCY";
    referral = "Immediate physician review; consider emergency department";
  } else if (score >= 45) {
    level = "HIGH";
    referral = "Same-day specialist referral (pulmonology / internal medicine)";
  } else if (score >= 20) {
    level = "MODERATE";
    referral = "Outpatient consultation within 48 hours";
  } else {
    level = "LOW";
    referral = "Self-care with safety-net advice";
  }
  if (!reasons.length) reasons.push("No concerning findings from symptoms or vitals");

  return { level, score, reasons, advisory, referral };
}

export const LEVEL_COLOR: Record<TriageLevel, string> = {
  LOW: "var(--ok)",
  MODERATE: "var(--warn)",
  HIGH: "var(--high)",
  EMERGENCY: "var(--danger)",
};

/** Map ticked evidence codes to the symptom names the triage rules use. */
export const CODE_TO_SYMPTOM: Record<string, string> = {
  E_66: "breathlessness",
  E_64: "breathlessness",
  E_14: "chest_pain",
  E_220: "chest_pain",
  E_45: "hemoptysis",
  E_159: "syncope",
  E_82: "syncope",
};
