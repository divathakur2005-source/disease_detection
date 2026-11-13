# Results

All numbers are on held-out **test** splits never used for training or model selection.

## 1. Clinical (tabular) models

| model | data | test rows | accuracy | macro-F1 | AUC | notes |
|---|---|---|---|---|---|---|
| Diabetes-risk MLP | Kaggle diabetes (100k) | 14,422 | 0.8914 | 0.7671 | 0.9721 | class-weighted; recall(diabetic) = 0.912 |
| Symptom→diagnosis MLP | DDXPlus (49 dx) | 134,529 | 0.9934 | 0.9922 | 0.9999 | top-3 acc 1.0000 |

Respiratory classes of the symptom model (test):

| diagnosis | precision | recall | F1 | support |
|---|---|---|---|---|
| Acute COPD exacerbation / infection | 1.000 | 1.000 | 1.000 | 2153 |
| Bronchitis | 0.999 | 0.997 | 0.998 | 3594 |
| Influenza | 1.000 | 1.000 | 1.000 | 3554 |
| Pneumonia | 1.000 | 1.000 | 1.000 | 3542 |
| Tuberculosis | 1.000 | 1.000 | 1.000 | 2080 |
| URTI | 0.999 | 0.993 | 0.996 | 8743 |
| Viral pharyngitis | 0.992 | 0.994 | 0.993 | 8334 |

SymptomEncoder pre-training (token encoder used by the fusion model) on DDXPlus: test accuracy 0.9933, macro-F1 0.9922.


## 2. Chest X-ray models (Open-I, 8 thoracic labels, same patient split)

| model | macro AUC | macro-F1 (tuned thr) | AUC normal | AUC chest_infection | AUC cardiomegaly | AUC pleural_effusion | AUC opacity | AUC atelectasis | AUC emphysema_copd | AUC edema_congestion |
|---|---|---|---|---|---|---|---|---|---|---|
| image | 0.826 | 0.407 | 0.819 | 0.833 | 0.922 | 0.867 | 0.656 | 0.796 | 0.888 | 0.830 |
| fusion_nopretrain | 0.825 | 0.407 | 0.780 | 0.833 | 0.916 | 0.861 | 0.663 | 0.775 | 0.885 | 0.886 |
| fusion | 0.813 | 0.374 | 0.799 | 0.807 | 0.907 | 0.835 | 0.621 | 0.818 | 0.884 | 0.835 |
| clinicaltext | 0.626 | 0.232 | 0.584 | 0.686 | 0.610 | 0.687 | 0.566 | 0.614 | 0.595 | 0.670 |
| clinical | 0.581 | 0.209 | 0.588 | 0.575 | 0.623 | 0.658 | 0.541 | 0.494 | 0.571 | 0.596 |

Test set: 466 patients; positives per label: normal 168, chest_infection 25, cardiomegaly 41, pleural_effusion 24, opacity 118, atelectasis 40, emphysema_copd 43, edema_congestion 7.


## 3. Symptom ↔ X-ray correlation (paired Open-I data)

Lift = P(X-ray label | symptom parsed from the indication) / P(X-ray label). Support n ≥ 20.

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


