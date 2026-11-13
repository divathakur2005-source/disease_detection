"use client";

import { useMemo, useState } from "react";
import type { Loaded } from "@/app/page";
import {
  CHEST_INFECTION_LIKE,
  COLD_LIKE,
  buildDiabetesInput,
  buildSymptomInput,
  forward,
  groupProbability,
  softmax,
} from "@/lib/model";
import { CODE_TO_SYMPTOM, LEVEL_COLOR, triage } from "@/lib/triage";

const SMOKING = ["never", "former", "current", "not current", "ever", "No Info"];

/**
 * Illustrative patients, so a reviewer can see the system behave without typing.
 * `note` explains what the case is meant to show - including where the model's
 * answer is interesting rather than simply "correct".
 */
const PRESETS: Record<string, {
  codes: string[]; age: number; male: boolean; bmi: number; hba1c: number;
  glucose: number; htn: boolean; hd: boolean; smoking: string; note: string;
}> = {
  "Mild upper-respiratory illness": {
    codes: ["E_201", "E_97", "E_181", "E_91"],
    age: 28, male: false, bmi: 23, hba1c: 5.2, glucose: 92, htn: false, hd: false, smoking: "never",
    note: "Cough, sore throat, blocked nose and fever in a young adult. The model should favour URTI and the triage should stay LOW.",
  },
  "Lower-respiratory infection": {
    codes: ["E_201", "E_77", "E_91", "E_94", "E_66", "E_14", "E_220", "E_89"],
    age: 67, male: true, bmi: 26, hba1c: 5.4, glucose: 100, htn: true, hd: false, smoking: "former",
    note: "Productive cough with fever, chills, breathlessness and pleuritic chest pain. The model puts most of the probability on a chest infection, though it spreads it across pneumonia, bronchitis and tuberculosis — which share almost all of these symptoms. Separating them is exactly what the chest X-ray branch is for.",
  },
  "Chest infection in a diabetic patient": {
    codes: ["E_201", "E_77", "E_91", "E_94", "E_66", "E_69", "E_162"],
    age: 58, male: true, bmi: 31, hba1c: 8.1, glucose: 210, htn: true, hd: false, smoking: "former",
    note: "The comorbidity case the project is about. The model leans strongly toward tuberculosis, because in DDXPlus diabetes and weight loss are TB risk factors rather than pneumonia ones. Either way the triage engine escalates, and the advisory flags the metabolic interaction.",
  },
  "Cardiac presentation": {
    codes: ["E_66", "E_14", "E_155", "E_151", "E_106", "E_104", "E_105"],
    age: 74, male: true, bmi: 29, hba1c: 5.9, glucose: 120, htn: true, hd: true, smoking: "former",
    note: "Breathlessness, palpitations and swelling with a cardiac history — the model should look away from infection entirely.",
  },
};

export default function Assess({ data }: { data: Loaded }) {
  const { ddx, ddxLayers, diab, diabLayers } = data;
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [age, setAge] = useState(45);
  const [male, setMale] = useState(false);
  const [bmi, setBmi] = useState(26);
  const [hba1c, setHba1c] = useState(5.6);
  const [glucose, setGlucose] = useState(110);
  const [htn, setHtn] = useState(false);
  const [hd, setHd] = useState(false);
  const [smoking, setSmoking] = useState("never");
  const [showAll, setShowAll] = useState(false);
  const [presetNote, setPresetNote] = useState<string | null>(null);

  const quick = ddx.quick_codes ?? [];
  const extra = useMemo(
    () => ddx.features.filter((f) => f.startsWith("E_") && !quick.includes(f)),
    [ddx.features, quick]
  );

  function toggle(code: string) {
    setTicked((prev) => {
      const next = new Set(prev);
      next.has(code) ? next.delete(code) : next.add(code);
      return next;
    });
  }

  function applyPreset(name: string) {
    const p = PRESETS[name];
    if (!p) return;
    setTicked(new Set(p.codes));
    setAge(p.age); setMale(p.male); setBmi(p.bmi); setHba1c(p.hba1c);
    setGlucose(p.glucose); setHtn(p.htn); setHd(p.hd); setSmoking(p.smoking);
    setPresetNote(p.note);
  }

  // Inference runs on every change - the models are small enough that this is instant.
  const out = useMemo(() => {
    const probs = softmax(forward(ddxLayers, buildSymptomInput(ddx, ticked, age, male)));
    const ranked = ddx.classes
      .map((c, i) => ({ name: c, p: probs[i], icd: ddx.icd10?.[c] ?? "" }))
      .sort((a, b) => b.p - a.p);
    const pCold = groupProbability(ddx.classes, probs, COLD_LIKE);
    const pChest = groupProbability(ddx.classes, probs, CHEST_INFECTION_LIKE);

    const dProbs = softmax(
      forward(diabLayers, buildDiabetesInput(diab, { age, bmi, hba1c, glucose, hypertension: htn, heartDisease: hd, male, smoking }))
    );
    const pDiab = dProbs[1];

    const symptoms = new Set<string>();
    ticked.forEach((c) => { if (CODE_TO_SYMPTOM[c]) symptoms.add(CODE_TO_SYMPTOM[c]); });

    return {
      ranked, pCold, pChest, pDiab,
      tri: triage({ chestInfection: pChest, diabetes: pDiab, commonCold: pCold, symptoms, age }),
    };
  }, [ticked, age, male, bmi, hba1c, glucose, htn, hd, smoking, ddx, ddxLayers, diab, diabLayers]);

  return (
    <>
      <div className="card" style={{ marginBottom: 18 }}>
        <h2>Try a sample patient</h2>
        <p className="sub">Loads a realistic symptom and vitals profile, then updates every prediction live.</p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Object.keys(PRESETS).map((name) => (
            <button key={name} className="ghost" onClick={() => applyPreset(name)}>{name}</button>
          ))}
          <button className="ghost" onClick={() => { setTicked(new Set()); setPresetNote(null); }}>
            Clear all
          </button>
        </div>
        {presetNote && <div className="note">{presetNote}</div>}
      </div>

      <div className="grid two">
        <div>
          <div className="card" style={{ marginBottom: 18 }}>
            <h2>Patient & symptoms</h2>
            <p className="sub">
              {ticked.size} of {ddx.features.filter((f) => f.startsWith("E_")).length} clinical
              questions answered yes
            </p>
            <div className="row" style={{ marginBottom: 14 }}>
              <label className="field">
                <span>Age</span>
                <input type="number" min={0} max={110} value={age} onChange={(e) => setAge(+e.target.value)} />
              </label>
              <label className="field">
                <span>Sex</span>
                <select value={male ? "Male" : "Female"} onChange={(e) => setMale(e.target.value === "Male")}>
                  <option>Female</option>
                  <option>Male</option>
                </select>
              </label>
            </div>
            <div className="checks">
              {(showAll ? [...quick, ...extra] : quick).map((code) => (
                <label key={code}>
                  <input type="checkbox" checked={ticked.has(code)} onChange={() => toggle(code)} />
                  <span>{ddx.questions?.[code] ?? code}</span>
                </label>
              ))}
            </div>
            <button className="ghost" style={{ marginTop: 12 }} onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show common questions only" : `Show all ${quick.length + extra.length} questions`}
            </button>
          </div>

          <div className="card">
            <h2>Vitals</h2>
            <p className="sub">Feeds the diabetes-risk model and the triage engine.</p>
            <div className="row three">
              <label className="field">
                <span>BMI</span>
                <input type="number" step="0.1" value={bmi} onChange={(e) => setBmi(+e.target.value)} />
              </label>
              <label className="field">
                <span>HbA1c (%)</span>
                <input type="number" step="0.1" value={hba1c} onChange={(e) => setHba1c(+e.target.value)} />
              </label>
              <label className="field">
                <span>Glucose (mg/dL)</span>
                <input type="number" value={glucose} onChange={(e) => setGlucose(+e.target.value)} />
              </label>
            </div>
            <label className="field">
              <span>Smoking history</span>
              <select value={smoking} onChange={(e) => setSmoking(e.target.value)}>
                {SMOKING.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <div style={{ display: "flex", gap: 18, fontSize: "0.86rem" }}>
              <label style={{ display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                <input type="checkbox" checked={htn} onChange={(e) => setHtn(e.target.checked)} /> Hypertension
              </label>
              <label style={{ display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                <input type="checkbox" checked={hd} onChange={(e) => setHd(e.target.checked)} /> Heart disease
              </label>
            </div>
          </div>
        </div>

        <div>
          <div
            className="card triage"
            style={{ borderColor: LEVEL_COLOR[out.tri.level], marginBottom: 18 }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span className="lvl" style={{ color: LEVEL_COLOR[out.tri.level] }}>
                {out.tri.level}
              </span>
              <span style={{ fontSize: "0.82rem", color: "var(--muted)" }}>
                urgency {out.tri.score}/100
              </span>
            </div>
            <div style={{ fontSize: "0.88rem", marginTop: 3 }}>{out.tri.referral}</div>
            <ul>
              {out.tri.reasons.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
            {out.tri.advisory.length > 0 && (
              <>
                <h3 style={{ marginBottom: 4 }}>Advisory</h3>
                <ul style={{ marginTop: 0 }}>
                  {out.tri.advisory.map((a, i) => <li key={i}>{a}</li>)}
                </ul>
              </>
            )}
          </div>

          <div className="card" style={{ marginBottom: 18 }}>
            <h2>Differential diagnosis</h2>
            <p className="sub">DDXPlus model · 223 clinical questions → 49 diagnoses</p>
            {out.ranked.slice(0, 6).map((r) => (
              <div className="pred" key={r.name}>
                <div className="top">
                  <span>{r.name}{r.icd && <span style={{ color: "var(--muted)" }}> · {r.icd}</span>}</span>
                  <b>{(r.p * 100).toFixed(1)}%</b>
                </div>
                <div className="bar"><div style={{ width: `${Math.max(r.p * 100, 0.5)}%` }} /></div>
              </div>
            ))}
          </div>

          <div className="card">
            <h2>Risk summary</h2>
            <div className="row three" style={{ marginTop: 10 }}>
              <div className="stat">
                <div className="n" style={{ color: "var(--accent)" }}>{Math.round(out.pCold * 100)}%</div>
                <div className="l">Common-cold-like</div>
              </div>
              <div className="stat">
                <div className="n" style={{ color: "var(--high)" }}>{Math.round(out.pChest * 100)}%</div>
                <div className="l">Chest infection</div>
              </div>
              <div className="stat">
                <div className="n" style={{ color: "var(--warn)" }}>{Math.round(out.pDiab * 100)}%</div>
                <div className="l">Diabetes risk</div>
              </div>
            </div>
            <div className="note">
              The chest-infection figure groups Pneumonia, Bronchitis, Bronchiectasis, Tuberculosis
              and COPD exacerbation — conditions that share nearly all their symptoms, so the
              clinical model can tell you <i>that</i> there is a lower-respiratory problem far more
              reliably than <i>which one</i>. Telling them apart is the job of the chest X-ray
              branch, which runs in the full Python system.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
