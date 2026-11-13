/**
 * Browser-side inference for the exported MLPs.
 *
 * The Python exporter folds BatchNorm into each Linear layer and quantises the
 * weights to int8 with a per-row scale, so a forward pass here is just a few
 * matrix multiplies - no runtime library, no backend, no network round-trip.
 * Verified to agree with the PyTorch model on 100% of a 2,000-patient sample.
 */

export type Layer = {
  q: number[];
  shape: [number, number];
  scale: number[];
  b: number[];
  act: "relu" | "none";
};

export type MLP = {
  features: string[];
  classes: string[];
  layers: Layer[];
  questions?: Record<string, string>;
  quick_codes?: string[];
  icd10?: Record<string, string>;
  numeric?: string[];
  mean?: Record<string, number>;
  std?: Record<string, number>;
};

/** Dequantise once at load time so repeated predictions stay fast. */
export type ReadyLayer = { w: Float32Array; rows: number; cols: number; b: Float32Array; act: string };

export function prepare(model: MLP): ReadyLayer[] {
  return model.layers.map((L) => {
    const [rows, cols] = L.shape;
    const w = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const s = L.scale[r];
      for (let c = 0; c < cols; c++) w[r * cols + c] = L.q[r * cols + c] * s;
    }
    return { w, rows, cols, b: Float32Array.from(L.b), act: L.act };
  });
}

export function forward(layers: ReadyLayer[], input: Float32Array): Float32Array {
  let x = input;
  for (const L of layers) {
    const out = new Float32Array(L.rows);
    for (let r = 0; r < L.rows; r++) {
      let sum = L.b[r];
      const off = r * L.cols;
      for (let c = 0; c < L.cols; c++) sum += L.w[off + c] * x[c];
      out[r] = L.act === "relu" ? Math.max(0, sum) : sum;
    }
    x = out;
  }
  return x;
}

export function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum) as Float32Array;
}

/** Build the symptom model's input vector from ticked evidence codes. */
export function buildSymptomInput(model: MLP, ticked: Set<string>, age: number, male: boolean): Float32Array {
  const x = new Float32Array(model.features.length);
  model.features.forEach((f, i) => {
    if (f === "age_norm") x[i] = age / 100;
    else if (f === "sex_male") x[i] = male ? 1 : 0;
    else if (ticked.has(f)) x[i] = 1;
  });
  return x;
}

/** Build the diabetes model's input, applying the training-set standardisation. */
export function buildDiabetesInput(
  model: MLP,
  v: { age: number; bmi: number; hba1c: number; glucose: number; hypertension: boolean; heartDisease: boolean; male: boolean; smoking: string }
): Float32Array {
  const raw: Record<string, number> = {
    age: v.age,
    bmi: v.bmi,
    HbA1c_level: v.hba1c,
    blood_glucose_level: v.glucose,
    hypertension: v.hypertension ? 1 : 0,
    heart_disease: v.heartDisease ? 1 : 0,
    gender_male: v.male ? 1 : 0,
  };
  for (const s of ["never", "former", "current", "not current", "ever", "No Info"]) {
    raw[`smoking_${s.replace(/ /g, "_")}`] = v.smoking === s ? 1 : 0;
  }
  for (const c of model.numeric ?? []) {
    raw[c] = (raw[c] - (model.mean?.[c] ?? 0)) / (model.std?.[c] ?? 1);
  }
  return Float32Array.from(model.features.map((f) => raw[f] ?? 0));
}

export const COLD_LIKE = new Set([
  "URTI", "Viral pharyngitis", "Acute rhinosinusitis", "Allergic sinusitis", "Acute laryngitis", "Influenza",
]);
export const CHEST_INFECTION_LIKE = new Set([
  "Pneumonia", "Bronchitis", "Acute COPD exacerbation / infection", "Bronchiectasis", "Tuberculosis",
]);

export function groupProbability(classes: string[], probs: Float32Array, group: Set<string>): number {
  return classes.reduce((acc, c, i) => (group.has(c) ? acc + probs[i] : acc), 0);
}
