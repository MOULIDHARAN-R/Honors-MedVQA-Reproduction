# Experiment: Image Perturbation & Reliance Ablation

## Overview
This experiment tests whether the trained models genuinely depend on image content by re-scoring the finished models (`checkpoints/<run>/split_k/last.pt`) across all 5 test splits under three image conditions:
1. **Real**: The true, unmodified medical image.
2. **Shuffled**: Every test image is replaced by a different test image (fixed permutation seed = split index).
3. **Blank**: An all-zero (dataset-mean) image tensor.

* **Objective**: If accuracy barely drops when an image is scrambled or blanked, the model is ignoring the image for that question group. If accuracy plummets, the model is strongly grounded in visual features.

---

## Results Table (Run 4 Across 5 Splits)

| Category / Question Group | Real Image | Shuffled Image | Blank Image | Blind Prior | Visual Reliance Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall** | **$57.65\%$** | $20.05\%$ | $18.85\%$ | $36.36\%$ | Heavy vision dependence; scoring collapses on shuffled images |
| **Plane** | **$78.18\%$** | $28.10\%$ | $30.32\%$ | $48.83\%$ | Drops by $\sim 50$ points; strong visual feature extraction |
| **Organ System** | **$73.11\%$** | $20.35\%$ | $9.12\%$ | $40.49\%$ | Drops by $\sim 53-64$ points; strong visual feature extraction |
| **Modality (Open)** | **$71.00\%$** | $11.23\%$ | $11.18\%$ | $40.81\%$ | Drops by $\sim 60$ points; strictly vision-driven |
| **Modality (Choice)** | **$74.70\%$** | $33.38\%$ | $53.47\%$ | $53.47\%$ | On blank image, model cleanly falls back to question prior ($53.5\%$) |
| **Abnormality** | **$12.73\%$** | $2.17\%$ | $2.38\%$ | $3.67\%$ | Scrambling drops accuracy below chance |
| **Modality (Yes/No)** | **$56.71\%$** | **$52.68\%$** | **$55.91\%$** | **$67.55\%$** | **Vision contributes only $+0.8\%$**; model fails to learn language prior |

---

## Key Scientific Findings
1. **True Visual Dependence**: On Plane, Organ System, and Open Modality questions, replacing the image drops performance by $50$ to $60$ percentage points. The model is not simply exploiting text shortcuts on these clinical categories.
2. **The Yes/No Pathology**:
   * Removing the image entirely (Blank) drops Yes/No accuracy by only $0.8$ points ($56.71\% \to 55.91\%$).
   * Even with no image, the model reaches only $55.91\%$, failing to predict the $67.55\%$ majority class.
   * This proves that for Yes/No questions, the multimodal network neither extracts useful image evidence nor leverages the language prior.

---

## How to Run

```powershell
python experiments/image_ablation/run_image_ablation.py --run run4
```
Results are saved to `experiments/image_ablation/results/image_ablation.csv` and `results/reproduction/run4/image_ablation.csv`.
