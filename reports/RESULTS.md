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
| fusion_nopretrain | 0.842 | 0.410 | 0.797 | 0.852 | 0.900 | 0.816 | 0.658 | 0.891 | 0.914 | 0.905 |
| image | 0.822 | 0.430 | 0.812 | 0.765 | 0.912 | 0.832 | 0.613 | 0.836 | 0.892 | 0.910 |
| fusion | 0.808 | 0.415 | 0.781 | 0.753 | 0.906 | 0.867 | 0.640 | 0.775 | 0.843 | 0.901 |
| clinical | 0.622 | 0.215 | 0.610 | 0.629 | 0.700 | 0.559 | 0.528 | 0.572 | 0.637 | 0.743 |

Test set: 519 patients; positives per label: normal 188, chest_infection 27, cardiomegaly 48, pleural_effusion 27, opacity 133, atelectasis 40, emphysema_copd 36, edema_congestion 17.


## 3. Symptom ↔ X-ray correlation (paired Open-I data)

Lift = P(X-ray label | symptom parsed from the indication) / P(X-ray label). Support n ≥ 20.

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


