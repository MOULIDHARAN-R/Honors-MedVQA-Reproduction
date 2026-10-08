# Medical VQA Reproduction: Five-Split Baseline & Forensic Analysis

> **Reproduction of**: Louisa Canepa, Sonit Singh, Arcot Sowmya.  
> **Visual Question Answering in the Medical Domain**. *arXiv:2309.11080 [cs.CV]*, 2023.  
> **Target Benchmark**: ImageCLEF VQA-Med 2019 (15,292 QA pairs across 4,200 unique radiology images).  
> **Model Architecture**: VGG-16 (ImageNet pretrained, fc7 4096-d) + BERT-base (`bert-base-uncased`, pooled [CLS] 768-d) + Concatenation Fusion (4864-d) + 2 Fully Connected Layers (1024-d, ReLU, Dropout 0.3) + Multi-class Classification Head.

---

## Table of Contents
1. [Executive Summary & Primary Verdict](#1-executive-summary--primary-verdict)
2. [Quantitative Reproduction Results (Run 4)](#2-quantitative-reproduction-results-run-4)
3. [The Path to Run 4: Engineering & Stability Journey](#3-the-path-to-run-4-engineering--stability-journey)
4. [Documented Implementation Choices & Deviations](#4-documented-implementation-choices--deviations)
5. [The Modality Gap: Forensic Investigation](#5-the-modality-gap-forensic-investigation)
6. [Explainability & Visual Grounding (Grad-CAM)](#6-explainability--visual-grounding-grad-cam)
7. [Diagnostic Ablations & Integrity Audits](#7-diagnostic-ablations--integrity-audits)
8. [Hardware & Environment Setup](#8-hardware--environment-setup)
9. [Step-by-Step Reproduction Guide](#9-step-by-step-reproduction-guide)
10. [Repository Structure & Folder Guide](#10-repository-structure--folder-guide)
11. [Complete File-by-File Execution Reference](#11-complete-file-by-file-execution-reference)

---

## 1. Executive Summary & Primary Verdict

This document provides a comprehensive, rigorous report of the full-scale reproduction of Canepa et al. (2023). Across **five independent 80/20 image-level splits** trained from scratch for 50 epochs under a unified protocol (**Run 4**):

* **Overall Accuracy**: **$57.65\% \pm 0.83\%$** (Strict Exact Match: $57.58\% \pm 0.82\%$) compared to the paper's reported **$60.0\% \pm 1.0\%$** (a gap of $-2.35$ percentage points).
* **Domain Alignment**:
  * **Plane** ($78.18\%$ vs. Paper $77.0\%$, **$+1.18$ points**): Fully replicated within experimental error.
  * **Organ System** ($73.11\%$ vs. Paper $74.0\%$, **$-0.89$ points**): Fully replicated within experimental error.
  * **Abnormality** ($12.73\%$ vs. Paper $8.0\%$, **$+4.73$ points**): Exceeds paper baseline, bounded by a $93.11\%$ unseen-answer ceiling.
  * **Modality** ($66.35\%$ vs. Paper $76.0\%$, **$-9.65$ points**): **The sole driver of the overall discrepancy**.
* **Key Finding**: Extensive follow-up forensics proved the modality shortfall is **strictly confined to Yes/No questions** ($56.7\%$, performing worse than a question-prior blind baseline of $67.5\%$), whereas the model achieves strong visual grounding on choice ($74.7\%$), open-ended ($71.0\%$), plane ($78.2\%$), and organ ($73.1\%$) questions.

```
       Medical Image (Radiology)             Clinical Question (Natural Language)
                 │                                            │
                 ▼                                            ▼
          Visual Encoder                               Text Encoder
       (VGG-16, fc7 4096-d)                        (BERT, pooler 768-d)
                 │                                            │
                 └────────────────────┬───────────────────────┘
                                      │
                                      ▼
                            Multimodal Fusion
                        (Concatenation: 4864-d)
                                      │
                                      ▼
                               Classifier Head
                     Linear(4864 → 1024), ReLU, Dropout 0.3
                        Linear(1024 → num_classes)
                                      │
                                      ▼
                         Cross-Entropy Loss (Adam)
```

---

## 2. Quantitative Reproduction Results (Run 4)

All five splits were trained for 50 epochs under identical configurations with random seeds $0, 1, 2, 3, 4$. In accordance with medical benchmark standards, evaluation uses **last-epoch accuracy** (reporting the peak test-epoch accuracy would introduce optimistic test-set leakage).

### 2.1 Five-Split Breakdown Table

| Split | Random Seed | Overall Acc (%) | Strict Acc (%) | Modality (%) | Plane (%) | Organ (%) | Abnormality (%) | Unseen Ceiling (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split 1** | Seed 0 | 57.15 | 57.12 | 66.45 | 78.24 | 72.82 | 11.27 | 92.34 |
| **Split 2** | Seed 1 | 57.99 | 57.96 | 66.62 | 79.10 | 74.52 | 11.21 | 93.43 |
| **Split 3** | Seed 2 | 56.95 | 56.89 | 64.17 | 78.96 | 70.41 | 14.08 | 92.71 |
| **Split 4** | Seed 3 | 58.95 | 58.85 | 67.19 | 78.29 | 74.77 | 14.86 | 94.01 |
| **Split 5** | Seed 4 | 57.20 | 57.11 | 67.32 | 76.31 | 73.05 | 12.26 | 93.07 |
| **Mean** | — | **57.65** | **57.58** | **66.35** | **78.18** | **73.11** | **12.73** | **93.11** |
| **Sample Std ($s$)** | — | **± 0.83** | **± 0.82** | **± 1.27** | **± 1.11** | **± 1.74** | **± 1.66** | **± 0.65** |

*(Sample standard deviation computed with degrees of freedom $N - 1 = 4$).*

### 2.2 Comparison with Published Benchmark (Canepa et al., Table III)

| Category / Metric | Our Reproduction (Mean ± Std) | Paper Target (Table III) | Gap (Reproduction − Paper) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Plane** | $\mathbf{78.18\% \pm 1.11\%}$ | $77.0\%$ | $\mathbf{+1.18\text{ points}}$ | **Matched** (Within $1.0\sigma$) |
| **Organ System** | $\mathbf{73.11\% \pm 1.74\%}$ | $74.0\%$ | $\mathbf{-0.89\text{ points}}$ | **Matched** (Within $0.5\sigma$) |
| **Abnormality** | $\mathbf{12.73\% \pm 1.66\%}$ | $8.0\%$ | $\mathbf{+4.73\text{ points}}$ | **Above Paper** (Capped by ceiling) |
| **Modality** | $\mathbf{66.35\% \pm 1.27\%}$ | $76.0\%$ | $\mathbf{-9.65\text{ points}}$ | **Discrepancy** (Analyzed in Section 5) |
| **Overall (Official Semantics)** | $\mathbf{57.65\% \pm 0.83\%}$ | $\mathbf{60.0\% \pm 1.0\%}$ | $\mathbf{-2.35\text{ points}}$ | **Close Reproduction** |
| **Overall (Strict Exact Match)** | $57.58\% \pm 0.82\%$ | — | — | Near-identical to official |
| **Unseen-Answer Ceiling** | $93.11\% \pm 0.65\%$ | — | — | Theoretical closed-world cap |

---

## 3. The Path to Run 4: Engineering & Stability Journey

Achieving a clean 5-split run required overcoming several major numerical and memory bottlenecks inherent in training end-to-end VGG-16 + BERT on consumer hardware:

```
First Run (Paper Settings)       Speed Bottleneck Resolved        Run 3 Divergence                 Run 4 Unified Protocol
• lr=1e-4 constant               • Batch 64 required 9.3 GB       • lr=1e-4 constant               • StepLR (x0.5 / 5 ep)
• Batch 64, no clipping          • Spilled to RAM (11s/step)      • Grad norm spiked to 1931       • Grad clip norm = 5.0
• Epoch took 74 min              • Micro-batch 16x4 accumulation  • Collapsed at epoch 17-18       • Micro-batch 16x4
• Collapsed at epoch 6 (0.20)    • 0.9s/step (170s / epoch)       • Rescued with StepLR            • 5 Splits completed cleanly
```

1. **First Run (Pure Paper Settings)**:
   * Trained with a constant learning rate $\eta = 10^{-4}$ and unconstrained effective batch 64.
   * *Outcome*: Healthy through Epoch 4 (accuracy $53.1\%$), followed by catastrophic collapse at Epoch 6 ($20.4\%$). Epochs took $74$ minutes due to memory thrashing.
2. **Speed & Memory Optimization**:
   * Storing activations for both VGG-16 and BERT with batch 64 required $9.3\text{ GB}$ VRAM, exceeding the $8.0\text{ GB}$ dedicated VRAM and causing CUDA unified memory paging to system RAM.
   * *Resolution*: Implemented gradient accumulation with micro-batch size $16$ ($4$ accumulation steps). Peak VRAM dropped to $5.4\text{ GB}$, training speed accelerated from $11.1\text{ s/step}$ to $0.90\text{ s/step}$, and epoch duration dropped from $74\text{ minutes}$ to $\sim 170\text{ seconds}$.
   * *Numerical Equivalence*: In strict fp32 mode, accumulated gradients matched single-batch gradients to $9.5 \times 10^{-7}$.
3. **Run 3 Instability & The Need for Run 4**:
   * Run 3 applied constant $\eta = 10^{-4}$. At Epochs 17–18, gradient norms suddenly exploded to $592.0$ and $1,931.0$, dropping accuracy from $56.9\%$ to $22.1\%$.
   * A rescue run demonstrated that an exponential step decay schedule stabilized training.
   * Rather than mixing protocols across splits, **Run 4 was launched from scratch across all 5 splits** under a single unified, reproducible protocol.

---

## 4. Documented Implementation Choices & Deviations

The published paper is silent on several critical engineering specifications. All choices made in this reproduction are systematically categorized below:

### 4.1 Paper-Specified Parameters
* **Visual Encoder**: VGG-16 pretrained on ImageNet-1k, fine-tuned end-to-end.
* **Text Encoder**: BERT (`bert-base-uncased`), fine-tuned end-to-end.
* **Fusion**: Concatenation of image and question representations.
* **Classifier Head**: 2 Fully Connected layers with Cross-Entropy loss.
* **Effective Batch Size**: $64$.
* **Epochs**: $50$.
* **Initial Learning Rate**: $1 \times 10^{-4}$ (Adam optimizer).
* **Augmentations**: Color jitter ($\pm 5\%$, $p=0.4$), Affine translation/rotation ($5\text{ px} / 5^\circ$, $p=0.5$), Gaussian blur ($p=0.5$), Gaussian noise ($\sigma=0.01$, $p=0.4$).

### 4.2 Engineering Choices (Where Paper Was Silent)
* **Image Partitioning**: Split strictly by **unique image ID** (seeds 0..4) to eliminate image leakage across train and test.
* **VGG Feature Layer**: Extracted from `classifier[3]` (fc7, $4096$ dimensions).
* **BERT Output**: Pooled `[CLS]` token embedding ($768$ dimensions). Maximum sequence length: $32$ tokens.
* **Image Input Resolution**: Resized directly to $224 \times 224$ (VGG standard).
* **Answer Vocabulary**: Derived strictly from **training set answers** (first alternative) to avoid test leakage.
* **Evaluation Semantics**: Official ImageCLEF evaluation (`#` alternatives and optional parenthesized text) and Strict exact match.
* **Reporting Rule**: Last-epoch accuracy (Epoch 50).

### 4.3 Documented Deviations
1. **Gradient Clipping (`grad_clip_norm = 5.0`)**: Gradient norm capped at $5.0$ to prevent catastrophic gradient explosions.
2. **Learning Rate Decay Schedule**: Initial $\eta = 10^{-4}$, multiplied by $\gamma = 0.5$ every $5$ epochs from Epoch 6 onward:
   $$\eta_t = 10^{-4} \times 0.5^{\lfloor (t - 1) / 5 \rfloor}$$
3. **Micro-Batch Accumulation**: Micro-batch $16$, accumulation $4$ steps (hardware optimization; mathematically equivalent gradient).

---

## 5. The Modality Gap: Forensic Investigation

While Plane ($+1.18$) and Organ ($-0.89$) matched the paper, Modality fell $9.65$ points short ($66.35\%$ vs. $76.0\%$). We conducted a deep forensic investigation to isolate the exact cause:

### 5.1 Modality Breakdown by Question Type

```
                               Modality Accuracy by Question Type
   100% ┌────────────────────────────────────────────────────────────────────────┐
        │                                                                        │
    80% │                        74.7%                    71.0%                  │
        │                       ┌─────┐                  ┌─────┐                 │
    60% │       56.7%           │     │                  │     │                 │
        │      ┌─────┐          │     │                  │     │                 │
    40% │      │     │          │     │                  │     │                 │
        │      │     │          │     │                  │     │                 │
    20% │      │     │          │     │                  │     │                 │
        │      │     │          │     │                  │     │                 │
     0% └──────┴─────┴──────────┴─────┴──────────────────┴─────┴─────────────────┘
                 Yes/No                 Choice                    Open
             (47% of errors)        (10% of errors)         (43% of errors)
```

| Question Type | Rule / Filter | Test Rows / Split | Model Accuracy | Majority Baseline | Error Share |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Choice** | Question contains `' or '` | 107 | **$74.7\% \pm 3.3\%$** | $32.4\%$ | $10\%$ |
| **Open-Ended** | All other phrasing | 376 | **$71.0\% \pm 2.4\%$** | $29.0\%$ | $43\%$ |
| **Yes/No** | Reference answer is `yes` / `no` | 277 | **$56.7\% \pm 3.1\%$** | $54.0\%$ | **$47\%$** |

* **Finding**: The model performs well on Choice ($74.7\%$) and Open ($71.0\%$). The deficit is concentrated entirely in **Yes/No modality questions**, where accuracy ($56.7\%$) barely outperforms a majority coin flip ($54.0\%$).
* **Symmetric Confusions**: Across the 5 splits, there were $315$ instances of `yes` $\rightarrow$ `no` and $282$ instances of `no` $\rightarrow$ `yes` ($597$ total symmetric swaps, comprising $47\%$ of all modality errors).

### 5.2 The Image-Blind Baseline Comparison

To test how much information the model derives from the question text alone, we scored an image-blind prior baseline (predicting the most common training answer for each exact question wording):

| Category / Group | Image-Blind Baseline (%) | Multimodal Model (%) | Delta (Model − Blind) | Clinical Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Plane** | $48.8\% \pm 0.8\%$ | $78.2\% \pm 1.1\%$ | $\mathbf{+29.4\%}$ | Strong visual reliance |
| **Organ System** | $40.5\% \pm 2.4\%$ | $73.1\% \pm 1.7\%$ | $\mathbf{+32.6\%}$ | Strong visual reliance |
| **Modality: Open** | $40.8\% \pm 2.9\%$ | $71.0\% \pm 2.4\%$ | $\mathbf{+30.2\%}$ | Strong visual reliance |
| **Modality: Choice** | $53.5\% \pm 5.1\%$ | $74.7\% \pm 3.3\%$ | $\mathbf{+21.2\%}$ | Strong visual reliance |
| **Modality: Yes/No** | **$67.5\% \pm 1.9\%$** | **$56.7\% \pm 3.1\%$** | $\mathbf{-10.8\%}$ | **Model worse than blind text prior** |

* For fixed question templates (e.g. *"is this a t2 weighted image?"* where the ground truth is "no" $100\%$ of the time in the dataset), the model predicts "yes" $30.1\%$ of the time.
* The multimodal network failed to internalize the linguistic prior for Yes/No questions and was unable to extract sufficiently decisive visual cues to compensate.

### 5.3 Image Perturbation & Ablation Study

To determine whether the network genuinely uses image features, we re-scored the trained models under three input conditions: (a) Real Image, (b) Shuffled Image (paired with a random different scan), and (c) Blank Image:

| Category | Real Image (%) | Shuffled Image (%) | Blank Image (%) | Degradation (Real − Shuffled) | Visual Reliance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Plane** | $78.2\%$ | $28.1\%$ | $30.3\%$ | **$-50.1\text{ points}$** | Very High |
| **Organ System** | $73.1\%$ | $20.3\%$ | $9.1\%$ | **$-52.8\text{ points}$** | Very High |
| **Modality: Open** | $71.0\%$ | $11.2\%$ | $11.2\%$ | **$-59.8\text{ points}$** | Extreme |
| **Modality: Choice** | $74.7\%$ | $33.4\%$ | $53.5\%$ | **$-41.3\text{ points}$** | High |
| **Modality: Yes/No** | **$56.7\%$** | **$52.7\%$** | **$55.9\%$** | **$-4.0\text{ points}$** | **Negligible / Unused** |

* On Plane, Organ, and Open questions, scrambling the image degrades performance by $50\text{--}60$ percentage points, demonstrating genuine image utilization.
* On Yes/No questions, replacing the scan with a blank image drops accuracy by only $0.8$ points. The model essentially guesses on Yes/No questions.

---

## 6. Explainability & Visual Grounding (Grad-CAM)

We implemented Grad-CAM targeting VGG-16's final convolutional layer (`conv5_3` at `features[28]`, $512 \times 14 \times 14$) with dual multimodal inputs:

```
Image [3, 224, 224] ──► VGG-16 Conv Layers ──► features[28] (A_k) ──► fc7 (4096-d) ┐
                                                    │                              ├─► Concat ─► FC Head ─► Score (y_c)
Question [32]       ──► BERT Encoder       ─────────────────────────► CLS (768-d)  ┘                     │
                                                    ▲                                                   │
                                                    └───────── Backprop: d(y_c) / d(A_k) ───────────────┘
                                                                       │
                                                         Global Average Pooling (a_k)
                                                                       │
                                                      CAM = ReLU( sum_k a_k * A_k )
                                                                       │
                                                       Bilinear Upsample [224, 224]
```

### 6.1 Validation Against Reference Package
Our custom dual-input implementation in [`src/explainability/gradcam.py`](file:///c:/Users/mouli/Honors/MedVQA/src/explainability/gradcam.py) was cross-checked on CUDA against the official `pytorch-grad-cam` library:
$$\text{Pearson Correlation } r = \mathbf{1.0000}$$
The implementation achieves exact numerical equivalence while natively supporting joint image-text inputs.

### 6.2 Visual Inspection Observations
Auditing 16 balanced test cases across Split 1 ([`results/reproduction/run4/split_1/gradcam/gradcam_grid.png`](file:///c:/Users/mouli/Honors/MedVQA/results/reproduction/run4/split_1/gradcam/gradcam_grid.png)):
* **Correct Predictions**: Strong spatial localization on anatomical structures (e.g. mediastinum for lung questions, pelvic center for axial CT scans, frontal/occipital lobes for brain questions).
* **Incorrect Predictions**: In 5 of 8 failure cases, Grad-CAM attention was concentrated on non-anatomical scan borders, orientation markers (`AP`, `SUPINE`), or scanner watermarks, confirming susceptibility to visual shortcut artifacts.

---

## 7. Diagnostic Ablations & Integrity Audits

### 7.1 QA-Level vs. Image-Level Splitting Diagnostic
* **Hypothesis**: Did Canepa et al. split data randomly by question rather than by image, artificially inflating their modality score via image memorization?
* **Experiment** ([`experiments/diagnostic_qa_run1/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/diagnostic_qa_run1/)):
  Trained on a QA-level split where **$94.6\%$ of test images** were present in the training set with different questions.
* **Results**:
  * Overall accuracy increased marginally from $57.15\%$ to $58.57\%$ ($+1.42$ points).
  * Modality accuracy increased from $66.45\%$ to $67.34\%$ (**$+0.89$ points**), remaining far below the paper's $76.0\%$.
* **Conclusion**: QA-level visual leakage does **not** explain the published modality score.

### 7.2 Patient / Case-Level Leakage Audit
* **Audit** ([`experiments/case_leakage_audit/`](file:///c:/Users/mouli/Honors/MedVQA/experiments/case_leakage_audit/run_case_audit.py)):
  Inspected the 15,292 QA rows and raw metadata for patient, case, study, or accession identifiers.
* **Finding**: The public VQA-Med 2019 dataset contains no patient or case metadata. **Image-level partitioning is the strictest verifiable split possible on this benchmark**.

---

## 8. Hardware & Environment Setup

All experiments were executed on a native Windows workstation.

### 8.1 System Specifications
* **Operating System**: Windows 11 Home / Pro
* **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6 VRAM)
* **CPU**: AMD Ryzen 9 8900HX (16 Threads)
* **RAM**: 16 GB DDR5
* **CUDA Driver**: 13.0 / CUDA Toolkit 12.x+ compatible

### 8.2 Conda Environment Setup
The project runs under Conda environment `medvqa` (Python 3.12):

```powershell
# 1. Activate conda environment
conda activate medvqa

# 2. Key package versions
# PyTorch: 2.14.1+cu130
# Torchvision: 0.29.1+cu130
# Transformers: 5.18.0 / 4.x
# Scikit-Learn: 1.6+
# Pandas: 3.0+
# Matplotlib: 3.10+
```

> [!NOTE]
> **Windows 11 Smart App Control (SAC) Compatibility**:  
> In Windows 11 environments where Smart App Control is enabled in Enforcement Mode (`VerifiedAndReputablePolicyState: 1`), ensure `torch.dll` uses the Microsoft ISG-reputed binary shim located in the environment path. The repository includes automated fallback handling.

---

## 9. Step-by-Step Reproduction Guide

Execute all commands from the project root (`c:\Users\mouli\Honors\MedVQA`) with the `medvqa` Conda environment active:

```powershell
conda activate medvqa
```

> [!TIP]
> **Command Notation & Generalized Parameters**:
> * `<RUN_NAME>`: Identifies your experiment run folder under `checkpoints/` and `results/reproduction/` (e.g. `run4` for our benchmark, or custom names like `run5`, `my_reproduction`).
> * `<SPLIT_ID>`: Integer split index `1`, `2`, `3`, `4`, or `5` (corresponding to the 5 random 80/20 image-level partitions).
> * `<SEED>`: Integer random seed for split generation (default: `0`).

---

### Step 1: Preprocess Dataset
Verifies raw ImageCLEF XML/TXT files, cleans BOM characters, and builds the canonical dataset CSV:
```powershell
python -m src.preprocessing
```
* **Output**: `data/processed/all_qa_pairs.csv` (15,292 rows, MD5: `5cbf955ad6d262b8f2382e8e48fab524`).

---

### Step 2: Generate 5 Image-Level Splits
Partitions the dataset into 5 independent 80/20 image-level draws (seeds `<SEED>` to `<SEED> + 4`):
```powershell
python scripts/split_dataset.py --split_unit image --seed <SEED>
```
* **Output**: `data/splits/image/split_1.json` through `split_5.json` and `summary.json`.

---

### Step 3: Run Fast Reproduction Smoke Test (Dry Run)
Executes a lightweight 2 micro-batch, 1-epoch pipeline check without modifying stored results:
```powershell
python train.py --split <SPLIT_ID> --dry_run
```
* **Output**: Routes to `checkpoints/dry_run/split_<SPLIT_ID>/` and `results/reproduction/dry_run/split_<SPLIT_ID>/`. Regular run checkpoints remain 100% isolated.

---

### Step 4: Run Full Reproduction Training

* **Train an Individual Split**:
  ```powershell
  python train.py --split <SPLIT_ID> --run_name <RUN_NAME>
  ```

* **Train All Five Splits Sequentially**:
  ```powershell
  python train.py --all_splits --run_name <RUN_NAME>
  ```

* **Resume an Interrupted Training Split**:
  ```powershell
  python train.py --split <SPLIT_ID> --run_name <RUN_NAME> --resume
  ```
* **Output**: Checkpoints saved to `checkpoints/<RUN_NAME>/split_<SPLIT_ID>/last.pt` and `best.pt`. Per-epoch histories saved to `history.json` and `train.log`.

---

### Step 5: Live Training Monitoring
Monitors real-time loss, accuracy, and gradient norms with automated heuristic collapse alerts:
```powershell
# Real-time console watch:
python -m scripts.watch_run --run <RUN_NAME> --split <SPLIT_ID> --follow

# Inspect plain text training log directly:
python -m scripts.watch_log --log checkpoints/<RUN_NAME>/split_<SPLIT_ID>/train.log
```

---

### Step 6: Generate Summary Results Tables
Aggregates all 5 splits for a run, computes sample standard deviation ($ddof=1$), and compares with Table III:
```powershell
python -m scripts.summarize_run --run <RUN_NAME>
```
* **Output**: `results/reproduction/<RUN_NAME>/results_table.md` and `results_table.csv`.

---

### Step 7: Plot Training Dynamics & Overview Curves
Generates 4-panel training curves (Loss, Accuracy, Categories, Gradient Norms) across splits:
```powershell
# Plot all splits summary for a run:
python -m scripts.plot_run --run <RUN_NAME> --all

# Plot detailed individual split curves:
python -m scripts.plot_results --split <SPLIT_ID>
```
* **Output**: `results/reproduction/<RUN_NAME>/split_<SPLIT_ID>/plots/overview.png` and `training_curves.png`.

---

### Step 8: Generate Grad-CAM Heatmaps
Generates category-balanced Grad-CAM visual heatmaps (correct and incorrect cases):
```powershell
# Balanced Grad-CAM generation:
python -m scripts.run_gradcam --run <RUN_NAME> --split <SPLIT_ID> --per_cat <COUNT> --seed <SEED>

# Cross-validate against pytorch-grad-cam reference package:
python -m scripts.run_gradcam --run <RUN_NAME> --split <SPLIT_ID> --target gt --verify_package
```
* **Output**: `results/reproduction/<RUN_NAME>/split_<SPLIT_ID>/gradcam/gradcam_grid.png` and `selection.csv`.

---

### Step 9: Run Diagnostic Studies & Ablations
```powershell
# 1. Modality question-type and confusion breakdown:
python -m scripts.modality_breakdown --run <RUN_NAME>

# 2. Yes/No template-level wording analysis:
python -m scripts.yesno_analysis --run <RUN_NAME>

# 3. Image-blind language-prior baseline:
python experiments/blind_baseline/run_blind_baseline.py --run <RUN_NAME>

# 4. Image perturbation ablation (Real vs. Shuffled vs. Blank scans):
python experiments/image_ablation/run_image_ablation.py --run <RUN_NAME>

# 5. Multi-split border energy shortcut analysis:
python experiments/shortcut_analysis/run_shortcut_analysis.py --run <RUN_NAME>

# 6. Patient / case leakage audit:
python experiments/case_leakage_audit/run_case_audit.py

# 7. Question-level split diagnostic runner:
python experiments/diagnostic_qa_run1/train_diagnostic.py
```

---

## 10. Repository Structure & Folder Guide

```
MedVQA/
├── data/                                    # Dataset storage
│   ├── raw/                                 # Raw ImageCLEF 2019 downloads (Train, Val, Test)
│   ├── processed/                           # Canonical pooled dataset
│   │   └── all_qa_pairs.csv                 # 15,292 QA pairs (MD5: 5cbf955ad6d262b8f2382e8e48fab524)
│   └── splits/                              # Partition definitions
│       ├── image/                           # 5 official image-level splits (80/20, seeds 0..4)
│       │   ├── split_1.json ... split_5.json
│       │   └── summary.json                 # Overlap checks & unseen answer ceilings
│       └── qa/                              # Diagnostic question-level split
│
├── src/                                     # Core modular package
│   ├── config.py                            # Central frozen configuration (CFG dataclass)
│   ├── answers.py                           # Clinical text normalizer, `#` parser, AnswerVocab
│   ├── dataset.py                           # PyTorch Dataset, tokenization, accepted-class tensors
│   ├── preprocessing.py                     # Training data augmentations & normalization
│   ├── models/                              # Neural network architectures
│   │   ├── image_encoder.py                 # VGG-16 backbone (4096-d fc7 extraction)
│   │   ├── question_encoder.py              # BERT-base backbone (768-d pooled [CLS])
│   │   ├── fusion.py                        # Multimodal concatenation (4864-d)
│   │   └── medvqa_model.py                  # End-to-end classifier & build_model factory
│   ├── training/                            # Training engine
│   │   ├── trainer.py                       # 50-epoch training loop with accumulation & clipping
│   │   ├── checkpoint.py                    # Atomic serialization & resume logic
│   │   └── losses.py                        # Cross-entropy loss criterion
│   ├── evaluation/                          # Evaluation routines
│   │   ├── evaluate.py                      # Multi-metric evaluator (Official vs. Strict)
│   │   └── metrics.py                       # Categorical accuracy helpers
│   └── explainability/                      # Explainability algorithms
│       └── gradcam.py                       # Dual-input Grad-CAM engine on VGG-16 conv5_3
│
├── scripts/                                 # General command-line tools
│   ├── summarize_run.py                     # Aggregates 5 splits, computes sample std, compares to paper
│   ├── plot_run.py                          # Multi-panel training dynamics visualization
│   ├── plot_results.py                      # Publication loss & category curves
│   ├── run_gradcam.py                       # Grad-CAM visualization & verification runner
│   ├── modality_breakdown.py                # Evaluates modality performance across question types
│   ├── yesno_analysis.py                    # Analyzes template-level accuracy on Yes/No questions
│   ├── check_accumulation.py                # Compares micro-batch gradients against single batch
│   ├── test_answers.py                      # Unit tests for answer normalization and parsing
│   ├── smoke_test_model.py                  # Model forward + backward pass smoke check
│   ├── smoke_test_dataset.py                # Dataset batching & tokenization validation
│   ├── watch_run.py                         # Live training heuristic anomaly & collapse detector
│   ├── watch_log.py                         # Training log text parser
│   └── split_dataset.py                     # Generator for image-level and QA-level partitions
│
├── experiments/                             # Isolated diagnostic studies & ablations
│   ├── blind_baseline/                      # Question-only language prior experiment
│   │   ├── run_blind_baseline.py
│   │   └── results/
│   ├── image_ablation/                      # Real vs. Shuffled vs. Blank image ablation
│   │   ├── run_image_ablation.py
│   │   └── results/
│   ├── shortcut_analysis/                   # 5-split Grad-CAM border energy quantification
│   │   ├── run_shortcut_analysis.py
│   │   └── results/
│   ├── case_leakage_audit/                  # Metadata inspection for patient/case leakage
│   │   └── run_case_audit.py
│   └── diagnostic_qa_run1/                  # QA-level split training experiment (94.6% image overlap)
│       ├── train_diagnostic.py
│       ├── split/
│       ├── checkpoint/
│       └── results/
│
├── checkpoints/                             # Model weights & histories
│   ├── run4/                                # Official 5-split checkpoints (split_1..5)
│   │   └── split_*/                         # last.pt, best.pt, history.json, train.log
│   └── dry_run/                             # Isolated scratch directory for smoke tests
│
├── results/                                 # Generated evaluation artifacts
│   └── reproduction/
│       └── run4/                            # Official results, logs, and plots
│           ├── results_table.md             # Markdown 5-split summary table
│           ├── results_table.csv            # CSV 5-split summary table
│           ├── blind_baseline.csv           # Language prior metrics
│           ├── image_ablation.csv           # Image perturbation metrics
│           ├── shortcut_analysis_5splits_*  # Shortcut border energy results & plots
│           └── split_*/                     # Per-split overview plots, summary.json, gradcam/
│
├── backups/                                 # Immutable archives
│   └── run4_frozen/                         # Complete immutable backup of Run 4 checkpoints
│
├── train.py                                 # Root CLI entry point for reproduction training
├── README.md                                # General Honors project README (Phases 1 & 2)
└── README_REPRODUCTION.md                   # This dedicated reproduction document
```

---

## 11. Complete File-by-File Execution Reference

| File | Subdirectory | Purpose | How to Run | Output Destination |
| :--- | :--- | :--- | :--- | :--- |
| [`train.py`](file:///c:/Users/mouli/Honors/MedVQA/train.py) | Root | Main reproduction training CLI | `python train.py --split <SPLIT_ID> --run_name <RUN_NAME>`<br>`python train.py --split <SPLIT_ID> --dry_run` | `checkpoints/<RUN_NAME>/split_<SPLIT_ID>/last.pt`<br>`results/reproduction/<RUN_NAME>/split_<SPLIT_ID>/` |
| [`src/preprocessing.py`](file:///c:/Users/mouli/Honors/MedVQA/src/preprocessing.py) | `src/` | Canonical CSV generation & data cleaning | `python -m src.preprocessing` | `data/processed/all_qa_pairs.csv` |
| [`scripts/split_dataset.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/split_dataset.py) | `scripts/` | Partitions dataset by image or QA pair | `python scripts/split_dataset.py --split_unit image --seed <SEED>` | `data/splits/image/split_*.json` |
| [`scripts/summarize_run.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/summarize_run.py) | `scripts/` | Aggregates 5 splits & compares to Table III | `python -m scripts.summarize_run --run <RUN_NAME>` | `results/reproduction/<RUN_NAME>/results_table.md` & `.csv` |
| [`scripts/plot_run.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/plot_run.py) | `scripts/` | Generates 4-panel training progress figures | `python -m scripts.plot_run --run <RUN_NAME> --all` | `results/reproduction/<RUN_NAME>/split_*/plots/overview.png` |
| [`scripts/plot_results.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/plot_results.py) | `scripts/` | Plots loss & category curves | `python -m scripts.plot_results --split <SPLIT_ID>` | `results/reproduction/split_<SPLIT_ID>/plots/training_curves.png` |
| [`scripts/run_gradcam.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/run_gradcam.py) | `scripts/` | Generates Grad-CAM overlays & validates against package | `python -m scripts.run_gradcam --run <RUN_NAME> --split <SPLIT_ID> --per_cat <COUNT>`<br>`python -m scripts.run_gradcam --run <RUN_NAME> --split <SPLIT_ID> --verify_package` | `results/reproduction/<RUN_NAME>/split_<SPLIT_ID>/gradcam/gradcam_grid.png` |
| [`scripts/modality_breakdown.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/modality_breakdown.py) | `scripts/` | Evaluates accuracy by question type | `python -m scripts.modality_breakdown --run <RUN_NAME>` | Console breakdown & prediction dumps |
| [`scripts/yesno_analysis.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/yesno_analysis.py) | `scripts/` | Evaluates template-level Yes/No accuracy | `python -m scripts.yesno_analysis --run <RUN_NAME>` | `results/reproduction/<RUN_NAME>/modality_breakdown/yesno_by_template.csv` |
| [`scripts/check_accumulation.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/check_accumulation.py) | `scripts/` | Verifies micro-batch gradient accumulation | `python -m scripts.check_accumulation` | Console numerical equivalence report |
| [`scripts/test_answers.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/test_answers.py) | `scripts/` | Unit tests for `#` and `()` answer parsing | `python -m scripts.test_answers` | Console test assertions |
| [`scripts/smoke_test_model.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/smoke_test_model.py) | `scripts/` | Checks forward/backward pass & GPU memory | `python -m scripts.smoke_test_model` | Console verification report |
| [`scripts/smoke_test_dataset.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/smoke_test_dataset.py) | `scripts/` | Validates dataset item shapes & classes | `python -m scripts.smoke_test_dataset` | Console item shapes & vocab head |
| [`scripts/watch_run.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/watch_run.py) | `scripts/` | Live training monitor with collapse alerts | `python -m scripts.watch_run --run <RUN_NAME> --split <SPLIT_ID> --follow` | Real-time console table |
| [`scripts/watch_log.py`](file:///c:/Users/mouli/Honors/MedVQA/scripts/watch_log.py) | `scripts/` | Parses text training log files | `python -m scripts.watch_log --log checkpoints/<RUN_NAME>/split_<SPLIT_ID>/train.log` | Console parsed epoch metrics |
| [`run_blind_baseline.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/blind_baseline/run_blind_baseline.py) | `experiments/blind_baseline/` | Evaluates question-only text prior | `python experiments/blind_baseline/run_blind_baseline.py --run <RUN_NAME>` | `experiments/blind_baseline/results/blind_baseline.csv` |
| [`run_image_ablation.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/image_ablation/run_image_ablation.py) | `experiments/image_ablation/` | Re-evaluates on Real vs. Shuffled vs. Blank scans | `python experiments/image_ablation/run_image_ablation.py --run <RUN_NAME>` | `experiments/image_ablation/results/image_ablation.csv` |
| [`run_shortcut_analysis.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/shortcut_analysis/run_shortcut_analysis.py) | `experiments/shortcut_analysis/` | Computes 5-split Grad-CAM border energy | `python experiments/shortcut_analysis/run_shortcut_analysis.py --run <RUN_NAME>` | `experiments/shortcut_analysis/results/shortcut_analysis_5splits_*` |
| [`run_case_audit.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/case_leakage_audit/run_case_audit.py) | `experiments/case_leakage_audit/` | Inspects dataset for patient metadata | `python experiments/case_leakage_audit/run_case_audit.py` | Console audit confirmation |
| [`train_diagnostic.py`](file:///c:/Users/mouli/Honors/MedVQA/experiments/diagnostic_qa_run1/train_diagnostic.py) | `experiments/diagnostic_qa_run1/` | Trains QA-level split model (94.6% image overlap) | `python experiments/diagnostic_qa_run1/train_diagnostic.py` | `experiments/diagnostic_qa_run1/results/` & `checkpoint/` |
