# Symptom ↔ X-ray correlation

Each row: a symptom or history item parsed from the referral indication, and how much more
likely the radiologist's finding becomes when it is present.
Lift = P(finding | symptom) / P(finding). Patients: 3,851; minimum support 20.

| symptom / history | X-ray finding | patients | base rate | given symptom | lift |
|---|---|---|---|---|---|
| Do you have heart failure? | edema_congestion | 21 | 2.4% | 19.0% | **8.0×** |
| Do you have heart failure? | pleural_effusion | 21 | 4.9% | 33.3% | **6.9×** |
| Hypoxia / low oxygen saturation | chest_infection | 35 | 5.3% | 25.7% | **4.8×** |
| Hypoxia / low oxygen saturation | pleural_effusion | 35 | 4.9% | 20.0% | **4.1×** |
| Do you have heart failure? | cardiomegaly | 21 | 9.5% | 33.3% | **3.5×** |
| Have you had surgery within the last month? | pleural_effusion | 55 | 4.9% | 16.4% | **3.4×** |
| Have you ever had pneumonia? | chest_infection | 81 | 5.3% | 16.0% | **3.0×** |
| Do you feel your heart is beating fast (racing), irregular | edema_congestion | 30 | 2.4% | 6.7% | **2.8×** |
| Do you feel your heart is beating fast (racing), irregular | pleural_effusion | 30 | 4.9% | 13.3% | **2.7×** |
| Do you have chronic kidney failure? | pleural_effusion | 46 | 4.9% | 13.0% | **2.7×** |
| Do you have a cough that produces colored or more abundant | chest_infection | 64 | 5.3% | 14.1% | **2.6×** |
| Line / tube / pacemaker placement check | pleural_effusion | 72 | 4.9% | 12.5% | **2.6×** |
| Follow-up or interval-comparison scan | pleural_effusion | 40 | 4.9% | 12.5% | **2.6×** |
| Are you experiencing shortness of breath or difficulty bre | edema_congestion | 676 | 2.4% | 5.9% | **2.5×** |
| Hypoxia / low oxygen saturation | cardiomegaly | 35 | 9.5% | 22.9% | **2.4×** |
| Hypoxia / low oxygen saturation | edema_congestion | 35 | 2.4% | 5.7% | **2.4×** |
| Do you have a cough? | chest_infection | 91 | 5.3% | 12.1% | **2.3×** |
| Do you feel your heart is beating fast (racing), irregular | cardiomegaly | 30 | 9.5% | 20.0% | **2.1×** |
| Have you ever had a heart attack or do you have angina (ch | cardiomegaly | 40 | 9.5% | 20.0% | **2.1×** |
| Have you ever had a heart attack or do you have angina (ch | pleural_effusion | 40 | 4.9% | 10.0% | **2.1×** |

These associations are clinically coherent - heart failure raising pulmonary edema and
pleural effusion, hypoxia raising chest infection, previous pneumonia raising it again -
which is what makes fusing the two modalities worthwhile rather than merely possible.

