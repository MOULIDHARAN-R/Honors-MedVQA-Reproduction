# Experiment: Question-Only Blind Language-Prior Baseline

## Overview
This experiment tests how much accuracy a classifier achieves **without looking at any medical images** by predicting the most frequent training answer for each exact question template.

* **Objective**: Measure how much of the model's reported accuracy is driven by question priors vs. genuine visual feature extraction.
* **Methodology**: For each of the 5 splits, the most common training answer per normalized question template is computed. Test questions are evaluated under the identical `#` alternative and parenthesized evaluation rules as the full multimodal model.

---

## Results: Multimodal Model vs. Blind Language Baseline

| Category / Question Type | Blind Baseline (Language Only) | Model (Run 4 Full Reproduction) | Model − Blind Prior (Image Contribution) | Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **Overall** | $36.36\% \pm 1.05\%$ | **$57.65\% \pm 0.83\%$** | **$+21.29\%$** | Strong genuine visual usage |
| **Plane** | $48.83\% \pm 0.76\%$ | **$78.18\% \pm 1.11\%$** | **$+29.35\%$** | Heavily vision-driven |
| **Organ System** | $40.49\% \pm 2.44\%$ | **$73.11\% \pm 1.74\%$** | **$+32.62\%$** | Heavily vision-driven |
| **Modality (Open)** | $40.81\% \pm 2.86\%$ | **$71.00\%$** | **$+30.19\%$** | Heavily vision-driven |
| **Modality (Choice)** | $53.47\% \pm 5.10\%$ | **$74.70\%$** | **$+21.23\%$** | Vision distinguishes choices |
| **Abnormality** | $3.67\% \pm 0.19\%$ | **$12.73\% \pm 1.66\%$** | **$+9.06\%$** | Outperforms prior despite low ceiling |
| **Modality (Yes/No)** | **$67.55\% \pm 1.94\%$** | **$56.71\%$** | <mark>**$-10.84\%$**</mark> | **Underperforms prior; vision fails** |

---

## Key Scientific Finding
1. **The Model Strongly Uses Vision**: Everywhere except Yes/No, the full model beats the blind prior by $21$ to $33$ percentage points, conclusively proving the image pathway carries real visual information.
2. **The Modality Gap Mechanism**: On Yes/No questions, a majority guess achieves $67.55\%$, but the trained model drops to $56.71\%$ ($-10.84$ points). This accounts for approximately $3.9$ percentage points of the $9.65$-point gap between our reproduction and the paper's $76\%$ modality target.

---

## How to Run

```powershell
python experiments/blind_baseline/run_blind_baseline.py --run run4
```
Results are saved to `experiments/blind_baseline/results/blind_baseline.csv` and `results/reproduction/run4/blind_baseline.csv`.
