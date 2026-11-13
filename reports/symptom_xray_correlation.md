# Symptom ↔ X-ray correlation

Each row: a symptom or history item parsed from the referral indication, and how much more
likely the radiologist's finding becomes when it is present.
Lift = P(finding | symptom) / P(finding). Patients: 3,105; minimum support 20.

| symptom / history | X-ray finding | patients | base rate | given symptom | lift |
|---|---|---|---|---|---|
| Hypoxia / low oxygen saturation | chest_infection | 27 | 5.4% | 25.9% | **4.8×** |
| Have you had surgery within the last month? | pleural_effusion | 42 | 4.6% | 19.0% | **4.2×** |
| Do you feel your heart is beating fast (racing), irregular | pleural_effusion | 25 | 4.6% | 16.0% | **3.5×** |
| Do you feel your heart is beating fast (racing), irregular | edema_congestion | 25 | 2.4% | 8.0% | **3.4×** |
| Hypoxia / low oxygen saturation | pleural_effusion | 27 | 4.6% | 14.8% | **3.2×** |
| Hypoxia / low oxygen saturation | edema_congestion | 27 | 2.4% | 7.4% | **3.2×** |
| Have you ever had pneumonia? | chest_infection | 60 | 5.4% | 16.7% | **3.1×** |
| Do you have chronic kidney failure? | pleural_effusion | 36 | 4.6% | 13.9% | **3.0×** |
| Do you have a cough that produces colored or more abundant | chest_infection | 50 | 5.4% | 16.0% | **3.0×** |
| Follow-up or interval-comparison scan | pleural_effusion | 32 | 4.6% | 12.5% | **2.7×** |
| Do you feel your heart is beating fast (racing), irregular | cardiomegaly | 25 | 9.0% | 24.0% | **2.7×** |
| Do you have a cough? | chest_infection | 71 | 5.4% | 14.1% | **2.6×** |
| Are you experiencing shortness of breath or difficulty bre | edema_congestion | 528 | 2.4% | 5.9% | **2.5×** |
| Do you have swollen or painful lymph nodes? | chest_infection | 30 | 5.4% | 13.3% | **2.5×** |
| Have you noticed any new fatigue, generalized and vague di | cardiomegaly | 27 | 9.0% | 22.2% | **2.5×** |
| Have you ever had a heart attack or do you have angina (ch | pleural_effusion | 28 | 4.6% | 10.7% | **2.3×** |
| Line / tube / pacemaker placement check | pleural_effusion | 47 | 4.6% | 10.6% | **2.3×** |
| Do you have chronic kidney failure? | cardiomegaly | 36 | 9.0% | 19.4% | **2.1×** |
| Have you ever had a heart attack or do you have angina (ch | cardiomegaly | 28 | 9.0% | 17.9% | **2.0×** |
| Are you experiencing shortness of breath or difficulty bre | cardiomegaly | 528 | 9.0% | 17.2% | **1.9×** |

These associations are clinically coherent - heart failure raising pulmonary edema and
pleural effusion, hypoxia raising chest infection, previous pneumonia raising it again -
which is what makes fusing the two modalities worthwhile rather than merely possible.

