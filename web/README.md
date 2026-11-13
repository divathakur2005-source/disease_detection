# Web demo — AI-Based Medical Decision-Support System

Interactive demo of the Major Project-I system (Group RG24, JUIT). Static
Next.js app: the trained models run **in the visitor's browser**, so there is
no backend, no server cost, and nothing to keep running.

## How the models get here

The PyTorch MLPs are exported by `scripts/export_web_models.py` in the project
root:

1. BatchNorm is folded into each preceding Linear layer, so a forward pass is
   just `y = Wx + b` plus ReLU.
2. Weights are quantised to int8 with a per-row scale — 5.2 MB → 901 KB.
3. The result is plain JSON in `public/models/`, read by `lib/model.ts`.

Verified against the PyTorch model on a 2,000-patient sample: **99.40%
accuracy in both, 100% agreement on predictions.**

The chest X-ray branch (DenseNet-121 + cross-modal attention) is too large to
ship to a browser, so this demo runs the two clinical models and the triage
engine. The radiograph pipeline runs in the Python system (`app/app.py`).

## Pages

| Tab | What it does |
|---|---|
| **Live assessment** | Tick symptoms and enter vitals → differential diagnosis across 49 conditions, diabetes risk, and a triage recommendation with written reasons. Four preset patients included. |
| **Results** | Measured test-set performance, the symptom ↔ X-ray correlation analysis, and dataset-linking coverage. |
| **How it works** | The three-dataset problem, the shared-vocabulary fix, the architecture, and the honest limitations. |

## Develop

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # static production build
```

To refresh the models and results after retraining:

```bash
cd ..
python scripts/export_web_models.py
```

## Deploy to Vercel

The app is a standard Next.js project with no server dependencies.

```bash
npm i -g vercel
vercel            # from this directory; accept the defaults
vercel --prod
```

Or connect the repository in the Vercel dashboard and set **Root Directory**
to `web`. No environment variables are required.

---

Research prototype for an academic project. Clinical decision *support*, not a
diagnosis; requires qualified medical review.
