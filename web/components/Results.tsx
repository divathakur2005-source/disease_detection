"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const LABEL_COLOR: Record<string, string> = {
  chest_infection: "#d1495b",
  cardiomegaly: "#edae49",
  pleural_effusion: "#00798c",
  edema_congestion: "#30638e",
};

const pct = (x: number) => `${(x * 100).toFixed(2)}%`;

export default function Results({ results }: { results: any }) {
  const { ddxplus, diabetes, coverage, correlation, clinical_openi } = results;
  const openi = ["openi_image", "openi_clinical", "openi_fusion", "openi_fusion_nopre"]
    .map((k) => ({ key: k, m: results[k] }))
    .filter((r) => r.m);

  const corrData = (correlation ?? []).slice(0, 10).map((r: any) => ({
    name: `${r.question.slice(0, 34)}${r.question.length > 34 ? "…" : ""} → ${r.label.replace(/_/g, " ")}`,
    lift: +r.lift.toFixed(2),
    n: r.n,
    label: r.label,
  })).reverse();

  return (
    <>
      <div className="card" style={{ marginBottom: 18 }}>
        <h2>Clinical models — held-out test performance</h2>
        <p className="sub">Evaluated on patients never seen during training or model selection.</p>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th className="num">Test patients</th>
              <th className="num">Accuracy</th>
              <th className="num">Macro-F1</th>
              <th className="num">AUC</th>
            </tr>
          </thead>
          <tbody>
            {ddxplus && (
              <tr>
                <td>Symptom → diagnosis (MLP)</td>
                <td>DDXPlus, 49 diagnoses</td>
                <td className="num">{ddxplus.n.toLocaleString()}</td>
                <td className="num">{pct(ddxplus.accuracy)}</td>
                <td className="num">{ddxplus.f1_macro.toFixed(3)}</td>
                <td className="num">{(ddxplus.auc_macro_ovr ?? 0).toFixed(4)}</td>
              </tr>
            )}
            {diabetes && (
              <tr>
                <td>Diabetes risk (MLP)</td>
                <td>Kaggle diabetes, 100k</td>
                <td className="num">{diabetes.n.toLocaleString()}</td>
                <td className="num">{pct(diabetes.accuracy)}</td>
                <td className="num">{diabetes.f1_macro.toFixed(3)}</td>
                <td className="num">{diabetes.auc.toFixed(4)}</td>
              </tr>
            )}
          </tbody>
        </table>
        {ddxplus?.per_class && (
          <>
            <h3>Respiratory diagnoses in detail</h3>
            <table>
              <thead>
                <tr>
                  <th>Diagnosis</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                  <th className="num">F1</th>
                  <th className="num">Cases</th>
                </tr>
              </thead>
              <tbody>
                {["Pneumonia", "Bronchitis", "URTI", "Influenza", "Tuberculosis"]
                  .filter((k) => ddxplus.per_class[k])
                  .map((k) => {
                    const v = ddxplus.per_class[k];
                    return (
                      <tr key={k}>
                        <td>{k}</td>
                        <td className="num">{v.precision.toFixed(3)}</td>
                        <td className="num">{v.recall.toFixed(3)}</td>
                        <td className="num">{v["f1-score"].toFixed(3)}</td>
                        <td className="num">{Math.round(v.support).toLocaleString()}</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </>
        )}
        <div className="note">
          DDXPlus is a synthetic corpus generated from a clinical knowledge base, so these scores
          are high by construction. They show the pipeline is correct; the honest test of the
          project is the chest X-ray fusion model below.
        </div>
      </div>

      {corrData.length > 0 && (
        <div className="card" style={{ marginBottom: 18 }}>
          <h2>Do symptoms actually predict X-ray findings?</h2>
          <p className="sub">
            This is the premise the whole project rests on. Lift = P(finding | symptom reported) ÷
            P(finding). Measured on the paired Open-I data; 1.0 means no association.
          </p>
          <ResponsiveContainer width="100%" height={corrData.length * 34 + 40}>
            <BarChart data={corrData} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }}>
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--muted)" }} />
              <YAxis
                type="category"
                dataKey="name"
                width={280}
                tick={{ fontSize: 10, fill: "var(--muted)" }}
              />
              <ReferenceLine x={1} stroke="var(--muted)" strokeDasharray="4 3" />
              <Tooltip
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--text)" }}
                formatter={(v: any, _n, p: any) => [`${v}× (n=${p.payload.n})`, "lift"]}
              />
              <Bar dataKey="lift" radius={[0, 4, 4, 0]} label={{ position: "right", fontSize: 10, fill: "var(--muted)", formatter: (v: any) => `${v}×` }}>
                {corrData.map((d: any, i: number) => (
                  <Cell key={i} fill={LABEL_COLOR[d.label] ?? "#0b3d91"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="note">
            These associations are clinically coherent — heart failure raising pulmonary edema and
            pleural effusion, hypoxia raising chest infection — which is what makes fusing the two
            modalities worthwhile rather than merely possible.
          </div>
        </div>
      )}

      <div className="grid two">
        {coverage && (
          <div className="card">
            <h2>Linking the datasets</h2>
            <p className="sub">
              How many chest X-ray referrals could be parsed into the shared clinical vocabulary.
            </p>
            <div className="row three" style={{ marginBottom: 12 }}>
              <div className="stat">
                <div className="n" style={{ color: "var(--ok)" }}>
                  {Math.round(coverage.parsed_any_pct * 100)}%
                </div>
                <div className="l">referrals parsed</div>
              </div>
              <div className="stat">
                <div className="n" style={{ color: "var(--accent)" }}>{coverage.vocab_size}</div>
                <div className="l">shared codes</div>
              </div>
              <div className="stat">
                <div className="n">{coverage.n_reports.toLocaleString()}</div>
                <div className="l">patients</div>
              </div>
            </div>
            <table>
              <tbody>
                <tr><td>At least one DDXPlus code</td><td className="num">{coverage.parsed_ddxplus.toLocaleString()}</td></tr>
                <tr><td>Unparsed referrals</td><td className="num">{coverage.unparsed.toLocaleString()}</td></tr>
                <tr><td>Mean codes per referral</td><td className="num">{coverage.mean_codes_per_report.toFixed(2)}</td></tr>
                <tr><td>Questions that ever fire</td><td className="num">{coverage.ddxplus_codes_used} / {coverage.n_ddxplus_codes}</td></tr>
              </tbody>
            </table>
            <div className="note">
              Referrals are short and de-identified (sensitive words are replaced with
              <code>XXXX</code>), so only the questions a radiology order actually mentions can
              fire. The limitation is reported rather than hidden.
            </div>
          </div>
        )}

        <div className="card">
          <h2>Chest X-ray models</h2>
          <p className="sub">
            All trained on the same patient split, so the comparison is fair. 8 thoracic findings,
            multi-label.
          </p>
          {openi.length > 0 ? (
            <table>
              <thead>
                <tr><th>Model</th><th className="num">Macro AUC</th><th className="num">Macro-F1</th></tr>
              </thead>
              <tbody>
                {openi.map(({ key, m }) => (
                  <tr key={key}>
                    <td>{key.replace("openi_", "").replace(/_/g, " ")}</td>
                    <td className="num">{(m["test_at_0.5"]?.auc_macro ?? 0).toFixed(3)}</td>
                    <td className="num">{(m.test_tuned?.f1_macro ?? 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <>
              {clinical_openi && (
                <table>
                  <thead>
                    <tr><th>Model</th><th className="num">Macro AUC</th><th className="num">Macro-F1</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>symptoms only (no image)</td>
                      <td className="num">{clinical_openi["test_at_0.5"].auc_macro.toFixed(3)}</td>
                      <td className="num">{clinical_openi.test_tuned.f1_macro.toFixed(3)}</td>
                    </tr>
                  </tbody>
                </table>
              )}
              <div className="note">
                Image-only, fusion and the pre-training ablation are still training. The
                symptoms-only score above is the lower bound the fusion model has to beat: if
                symptoms alone already matched it, the image branch would be adding nothing.
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
