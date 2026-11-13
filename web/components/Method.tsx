"use client";

export default function Method({ results }: { results: any }) {
  const cov = results.coverage;

  return (
    <>
      <div className="card" style={{ marginBottom: 18 }}>
        <h2>The problem</h2>
        <p style={{ marginTop: 6 }}>
          The project uses three public datasets — a symptom questionnaire corpus, a chest X-ray
          collection, and a diabetes records table. They describe <b>three different sets of
          patients</b>; nobody appears in more than one. So there was nothing linking a patient&apos;s
          reported symptoms to their radiograph, and &ldquo;multimodal fusion&rdquo; had nothing to fuse.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h2>The fix: the pairing was already inside the X-ray dataset</h2>
        <p style={{ marginTop: 6, marginBottom: 14 }}>
          Every Open-I radiology report carries an <b>indication</b> field — the referring
          clinician&apos;s reason for ordering the scan. That is the same patient&apos;s symptoms, sitting
          next to their image and the radiologist&apos;s verdict.
        </p>
        <table>
          <thead>
            <tr><th>Field in the report</th><th>What it contains</th><th>Role</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><code>indication</code></td>
              <td><i>&ldquo;XXXX-year-old female with dyspnea&rdquo;</i></td>
              <td>symptoms (modality A)</td>
            </tr>
            <tr>
              <td>image</td>
              <td>frontal chest radiograph</td>
              <td>image (modality B)</td>
            </tr>
            <tr>
              <td><code>MeSH</code></td>
              <td><i>&ldquo;Opacity/lung/base/left&rdquo;</i></td>
              <td>ground-truth label</td>
            </tr>
          </tbody>
        </table>
        <div className="note">
          The indication is free text, which is why it had been overlooked. Parsing it into
          structured clinical codes turns Open-I into genuinely paired multimodal data.
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h2>One shared clinical vocabulary</h2>
        <p style={{ marginTop: 6, marginBottom: 14 }}>
          The symptom dataset turned out to be <b>DDXPlus</b> — 610,480 patients answering 223
          standardised clinical questions, labelled with 49 diagnoses. Those question codes become
          the vocabulary that every data source speaks:
        </p>
        <div className="flow">
          <div className="step">
            <b>Open-I referral text</b>
            <i>&ldquo;XXXX-year-old female with dyspnea&rdquo;</i>
          </div>
          <div className="arrow">↓ pattern matching</div>
          <div className="step">
            <b>Shared code E_66</b>
            &ldquo;Are you experiencing shortness of breath or difficulty breathing?&rdquo; — the exact
            DDXPlus question
          </div>
          <div className="arrow">↓</div>
          <div className="step">
            <b>Both datasets now share a feature space</b>
            {cov ? `${cov.vocab_size} codes = ${cov.n_ddxplus_codes} DDXPlus questions + ${cov.n_extra_codes} Open-I-specific extras` : "223 DDXPlus questions plus Open-I-specific extras"}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h2>Why DDXPlus is still needed</h2>
        <p style={{ marginTop: 6 }}>
          Open-I has only ~3,100 usable patients — too few to train a symptom encoder well.
          DDXPlus has 610,480. So the encoder is <b>pre-trained</b> on DDXPlus and transferred into
          the fusion model, exactly as DenseNet-121 is pre-trained on ImageNet before being
          fine-tuned on radiographs. The shared vocabulary is what makes that transfer possible:
          index 66 means the same thing in both datasets.
        </p>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <h2>System architecture</h2>
        <div className="flow" style={{ marginTop: 10 }}>
          <div className="step">
            <b>Clinical branch</b>
            SymptomEncoder — one learned embedding per reported symptom, pre-trained on 610k DDXPlus patients
          </div>
          <div className="step">
            <b>Imaging branch</b>
            DenseNet-121 (ImageNet-initialised) → 7×7×1024 spatial feature map
          </div>
          <div className="arrow">↓</div>
          <div className="step">
            <b>Bidirectional cross-modal attention</b>
            Each symptom attends over the lung regions that could explain it; the image summary
            attends back over the reported symptoms. 9.0M parameters.
          </div>
          <div className="arrow">↓</div>
          <div className="step">
            <b>Multi-label head — 8 thoracic findings</b>
            normal · chest infection · cardiomegaly · pleural effusion · opacity · atelectasis ·
            emphysema/COPD · pulmonary edema
          </div>
          <div className="arrow">↓</div>
          <div className="step">
            <b>Grad-CAM + rule-based triage</b>
            Heat-map over the radiograph, then LOW / MODERATE / HIGH / EMERGENCY with written reasons
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Honest limitations</h2>
        <ul style={{ marginTop: 8, paddingLeft: 20, fontSize: "0.9rem" }}>
          <li style={{ marginBottom: 6 }}>
            <b>Referrals are short.</b> A clinician names one or two symptoms, not a full
            questionnaire — about {cov ? cov.mean_codes_per_report.toFixed(2) : "1.3"} codes per
            patient on average.
          </li>
          <li style={{ marginBottom: 6 }}>
            <b>De-identification removes words.</b> Open-I replaces sensitive tokens with
            <code>XXXX</code>, so some common symptoms are systematically under-counted.
          </li>
          <li style={{ marginBottom: 6 }}>
            <b>Only {cov ? cov.ddxplus_codes_used : 44} of 223 questions ever fire</b> — a chest
            X-ray order never mentions rash colour or menstrual history.
          </li>
          <li style={{ marginBottom: 6 }}>
            <b>DDXPlus is synthetic.</b> Its 99% accuracy reflects a clean generated corpus, not
            real-world difficulty.
          </li>
          <li>
            <b>This demo runs the clinical models only.</b> The chest X-ray branch needs a
            convolutional network too large for a browser; it runs in the full Python system.
          </li>
        </ul>
      </div>
    </>
  );
}
