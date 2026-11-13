# Indication parsing coverage

The project links symptoms to radiographs by parsing each Open-I report's free-text
*indication* (the referring clinician's reason for the scan) into the shared DDXPlus
evidence vocabulary. This file audits how well that works.

- Reports: **3,851** (3,765 have indication text)
- Vocabulary: **233** codes = 223 DDXPlus + 10 Open-I extras
- At least one code parsed: **3,183 (83%)**
- At least one *DDXPlus* code: **2,431 (63%)**
- Unparsed: **668** (of which 258 are empty or only de-identified tokens)
- Mean codes per report: **1.29**
- DDXPlus questions that ever fire: **44 of 223**

## Most frequent parsed codes

| code | meaning | reports |
|---|---|---|
| `E_14` | Do you have chest pain even at rest? | 903 |
| `E_66` | Are you experiencing shortness of breath or difficulty breathing in a  | 676 |
| `X_FEMALE` | Female patient | 676 |
| `X_MALE` | Male patient | 470 |
| `E_34` | Do you have an active cancer? | 211 |
| `X_PREOP` | Pre-operative / clearance screening | 199 |
| `X_PAIN_OTHER` | Pain outside the chest (back, abdomen, shoulder, neck) | 157 |
| `X_TRAUMA` | Trauma, fall or accident | 115 |
| `X_RULE_OUT` | Scan ordered to rule out / evaluate a suspected finding | 110 |
| `X_TB_SCREEN` | Tuberculosis screening (positive TB test / PPD) | 100 |
| `E_201` | Do you have a cough? | 91 |
| `E_227` | Are you immunosuppressed? | 83 |
| `E_118` | Have you ever had pneumonia? | 81 |
| `X_DEVICE` | Line / tube / pacemaker placement check | 72 |
| `E_79` | Do you smoke cigarettes? | 68 |
| `E_77` | Do you have a cough that produces colored or more abundant sputum than | 64 |
| `E_104` | Do you have high blood pressure or do you take medications to treat hi | 61 |
| `E_159` | Did you lose consciousness? | 57 |
| `E_123` | Do you have a chronic obstructive pulmonary disease (COPD)? | 56 |
| `E_196` | Have you had surgery within the last month? | 55 |
| `E_124` | Do you have asthma or have you ever had to use a bronchodilator in the | 48 |
| `E_113` | Do you have chronic kidney failure? | 46 |
| `E_148` | Are you feeling nauseous or do you feel like vomiting? | 44 |
| `E_105` | Have you ever had a heart attack or do you have angina (chest pain)? | 40 |
| `X_FOLLOWUP` | Follow-up or interval-comparison scan | 40 |

## Known limitations

1. **Indications are short.** A referral names one or two symptoms, not a full questionnaire,
   hence 1.29 codes per patient on average.
2. **De-identification removes words.** Open-I replaces sensitive tokens with `XXXX`
   (*"XXXX on the congestion, XXXX for 2 days"*), so some common symptoms are
   systematically under-counted.
3. **Only the questions a radiology referral mentions can fire** - the remaining DDXPlus
   questions (rash colour, menstrual history, travel) never appear in a chest X-ray order.

These are properties of the data, not defects of the method: the correlation analysis
(`symptom_xray_correlation.md`) shows the codes that *do* parse carry real diagnostic signal.

## Most common words in reports we fail to parse

| word | count |
|---|---|
| pain | 62 |
| history | 25 |
| loss | 20 |
| chest | 16 |
| chronic | 15 |
| weeks | 14 |
| one | 12 |
| right | 10 |
| left | 8 |
| pt | 7 |
| days | 7 |
| symptoms | 6 |
| ago | 5 |
| nodule | 5 |
| lower | 5 |
| post | 5 |
| months | 4 |
| arm | 4 |
| thoracic | 4 |
| side | 4 |
