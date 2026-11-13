# Symptom ↔ X-ray correlation

Each row: a symptom or history item parsed from the referral indication, and how much more
likely the radiologist's finding becomes when it is present.
Lift = P(finding | symptom) / P(finding). Patients: 3,459; minimum support 20.

| symptom / history | X-ray finding | patients | base rate | given symptom | lift |
|---|---|---|---|---|---|
| Hypoxia / low oxygen saturation | chest_infection | 29 | 5.3% | 24.1% | **4.6×** |
| Have you had surgery within the last month? | pleural_effusion | 47 | 4.6% | 17.0% | **3.7×** |
| Do you feel your heart is beating fast (racing), irregular | pleural_effusion | 25 | 4.6% | 16.0% | **3.5×** |
| Do you feel your heart is beating fast (racing), irregular | edema_congestion | 25 | 2.4% | 8.0% | **3.3×** |
| Have you ever had pneumonia? | chest_infection | 73 | 5.3% | 16.4% | **3.1×** |
| Do you have chronic kidney failure? | pleural_effusion | 42 | 4.6% | 14.3% | **3.1×** |
| Do you have a cough that produces colored or more abundant | chest_infection | 56 | 5.3% | 16.1% | **3.1×** |
| Hypoxia / low oxygen saturation | pleural_effusion | 29 | 4.6% | 13.8% | **3.0×** |
| Follow-up or interval-comparison scan | pleural_effusion | 37 | 4.6% | 13.5% | **2.9×** |
| Hypoxia / low oxygen saturation | edema_congestion | 29 | 2.4% | 6.9% | **2.9×** |
| Do you feel your heart is beating fast (racing), irregular | cardiomegaly | 25 | 9.0% | 24.0% | **2.7×** |
| Have you ever had a heart attack or do you have angina (ch | pleural_effusion | 33 | 4.6% | 12.1% | **2.6×** |
| Do you have a cough? | chest_infection | 80 | 5.3% | 13.8% | **2.6×** |
| Line / tube / pacemaker placement check | pleural_effusion | 51 | 4.6% | 11.8% | **2.5×** |
| Are you experiencing shortness of breath or difficulty bre | edema_congestion | 607 | 2.4% | 6.1% | **2.5×** |
| Do you have swollen or painful lymph nodes? | chest_infection | 32 | 5.3% | 12.5% | **2.4×** |
| Have you felt confused or disorientated lately? | pleural_effusion | 20 | 4.6% | 10.0% | **2.2×** |
| Do you have chronic kidney failure? | cardiomegaly | 42 | 9.0% | 19.0% | **2.1×** |
| Follow-up or interval-comparison scan | chest_infection | 37 | 5.3% | 10.8% | **2.1×** |
| Do you have chronic kidney failure? | edema_congestion | 42 | 2.4% | 4.8% | **2.0×** |

These associations are clinically coherent - heart failure raising pulmonary edema and
pleural effusion, hypoxia raising chest infection, previous pneumonia raising it again -
which is what makes fusing the two modalities worthwhile rather than merely possible.

