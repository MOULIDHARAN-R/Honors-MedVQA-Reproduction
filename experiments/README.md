# Experiments Directory Index

This directory organizes all **diagnostic ablations, representation studies, explainability audits, and baseline comparisons** conducted to dissect and validate the Med-VQA models.

> **Note**: The official 5-split baseline reproduction is implemented directly in [`src/`](file:///c:/Users/mouli/Honors/MedVQA/src/), launched via [`train.py`](file:///c:/Users/mouli/Honors/MedVQA/train.py), and evaluated via [`scripts/`](file:///c:/Users/mouli/Honors/MedVQA/scripts/). This directory is dedicated strictly to targeted diagnostic experiments and ablation studies.

---

## Directory Overview

| Experiment Folder | Purpose / Hypothesis Tested | Primary Runner | Key Artifacts |
| :--- | :--- | :--- | :--- |
| [`diagnostic_qa_run1/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/diagnostic_qa_run1/) | **QA-Level Split Diagnostic Ablation**: Tests whether question-level partitioning (introducing 94.6% image leakage) inflates accuracy through visual memorization. | [`train_diagnostic.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/diagnostic_qa_run1/train_diagnostic.py) | `split/split_1.json`, `checkpoint/last.pt`, live `overview.png` |
| [`blind_baseline/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/blind_baseline/) | **Language-Prior Blind Baseline**: Measures accuracy without medical images to identify question template bias and the Yes/No shortfall. | [`run_blind_baseline.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/blind_baseline/run_blind_baseline.py) | `results/blind_baseline.csv` (Overall $36.36\%$; Yes/No $67.55\%$) |
| [`image_ablation/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/image_ablation/) | **Image Perturbation & Reliance Ablation**: Re-scores models with Real vs. Shuffled vs. Blank images to prove visual grounding and isolate Yes/No failure. | [`run_image_ablation.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/image_ablation/run_image_ablation.py) | `results/image_ablation.csv` (Shuffled drops Plane to $28\%$, Organ to $20\%$) |
| [`shortcut_analysis/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/shortcut_analysis/) | **5-Split Border Energy Shortcut Audit**: Quantifies Grad-CAM activation energy in scan borders/corners vs. central anatomy. | [`run_shortcut_analysis.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/shortcut_analysis/run_shortcut_analysis.py) | `results/shortcut_analysis_5splits_table.csv`, plot png |
| [`case_leakage_audit/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/case_leakage_audit/) | **Patient / Case ID Identifier Audit**: Investigates whether raw dataset contains patient IDs to allow patient-level partitioning. | [`run_case_audit.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/case_leakage_audit/run_case_audit.py) | Confirms no patient/case metadata exists in VQA-Med 2019 |

---

## How to Run Experiments

All runner scripts include automatic project-root discovery:
```powershell
# 1. Run language-prior blind baseline:
python experiments/blind_baseline/run_blind_baseline.py --run run4

# 2. Run image perturbation ablation (Real vs Shuffled vs Blank):
python experiments/image_ablation/run_image_ablation.py --run run4

# 3. Run 5-split border energy shortcut audit:
python experiments/shortcut_analysis/run_shortcut_analysis.py --run run4

# 4. Run case identifier audit:
python experiments/case_leakage_audit/run_case_audit.py

# 5. Resume or run QA-split diagnostic:
python experiments/diagnostic_qa_run1/train_diagnostic.py --resume
```
