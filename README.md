# Honors-MedVQA-Reproduction

## Medical Visual Question Answering with VGG-16, BERT, and JEPA

This repository contains the implementation, reproduction, and research extensions for an Honors research project on **Medical Visual Question Answering (Med-VQA)**.

The project is structured into two major research phases:

1. **Phase 1 — Faithful Reproduction of the Published Med-VQA Baseline**: Re-implementing and empirically verifying the multimodal architecture proposed by Canepa et al. (*Visual Question Answering in the Medical Domain*) on the VQA-Med 2019 benchmark across five controlled image-level splits.
2. **Phase 2 — Vision-Language JEPA Extension**: Investigating the integration of Joint Embedding Predictive Architecture (JEPA / VL-JEPA) representations into medical VQA to evaluate whether predictive representation learning in continuous embedding space improves clinical accuracy, visual grounding, and robustness over traditional cross-entropy classification.

> [!IMPORTANT]
> **Dedicated Reproduction Report**: For an exhaustive, self-contained forensic report of Phase 1 (covering all 5 splits of Run 4, the modality gap forensics, Grad-CAM package cross-check, ablation studies, and step-by-step reproduction commands), see [**`README_REPRODUCTION.md`**](file:///c:/Users/mouli/Honors/MedVQA/README_REPRODUCTION.md).

---

# 1. Project Motivation

Medical Visual Question Answering requires a system to process a clinical radiology image (CT, MRI, X-ray, Ultrasound) alongside a natural-language clinical question and output a correct medical answer.

```
       Medical Image (Radiology)             Clinical Question (Natural Language)
                 │                                            │
                 ▼                                            ▼
          Visual Encoder                               Text Encoder
              (VGG-16)                                    (BERT)
                 │                                            │
                 └────────────────────┬───────────────────────┘
                                      │
                                      ▼
                            Multimodal Fusion
                           (Concatenation Layer)
                                      │
                                      ▼
                              Classifier Head
                           (2 Fully Connected Layers)
                                      │
                                      ▼
                               Clinical Answer
```

Medical VQA presents unique challenges compared to general-domain VQA:
* **High Inter-Class Similarity**: Minor anatomical variations or subtle pathology borders distinguish diagnostic classes.
* **Specialized Clinical Vocabulary**: Medical terms require precise semantic matching and domain-specific normalization.
* **Shortcut Learning & Dataset Biases**: Standard convolutional models often exploit background artifacts, scan borders, or superficial question correlations rather than clinical pathology.

This project first establishes a rigorous, reproducible baseline with detailed diagnostic audits, and then investigates a JEPA-based representation learning alternative.

---

# 2. Research Questions

* **RQ1 — Baseline Reproducibility**: Can the architecture proposed by Canepa et al. (VGG-16 + BERT + Concatenation + 2 FC layers) be reproduced on VQA-Med 2019 to match its reported performance (~60% accuracy)? What are the necessary implementation choices and deviations required for numerical stability?
* **RQ2 — Split Granularity & Data Leakage**: Does patient-level or image-level splitting vs. question-level splitting materially impact measured accuracy, and does question-level splitting introduce optimistic evaluation bias through visual memorization?
* **RQ3 — Visual Grounding & Shortcut Reliance**: Do the model's visual attention heatmaps (Grad-CAM) localize genuine anatomical pathology, or does the network rely on high-frequency corner/border shortcuts in radiology scans?
* **RQ4 — Vision-Language JEPA Adaptation**: Can a Joint Embedding Predictive Architecture (VL-JEPA) continuous embedding objective improve representation quality, generalization on rare medical conditions, and clinical visual grounding over the baseline?

---

# 3. Overall Research Pipeline

```
       ┌────────────────────────────────────────────────────────┐
       │                   VQA-Med 2019 Dataset                 │
       │       (4,200 images, 15,292 QA pairs, 4 categories)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              Phase 1: Baseline Reproduction            │
       │  • 5 Image-Disjoint 80/20 Splits (Seeds 0..4)          │
       │  • VGG-16 (fc7 4096-d) + BERT-base (pooler 768-d)      │
       │  • Concatenation Fusion -> 2 FC Layers (1024-d, Drop)  │
       │  • Adam (lr=1e-4, sched x0.5/5ep), Grad Clip = 5.0     │
       │  • Result: 57.65% ± 0.83% (Paper: 60.0% ± 1.0%)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│     Diagnostic Ablations        │ │      Explainability Audits      │
│ • QA-level vs. Image-level split│ │ • Grad-CAM Layer-28 Analysis    │
│ • Image leakage quantification  │ │ • Border energy shortcut metrics│
└────────────────┬────────────────┘ └────────────────┬────────────────┘
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            Phase 2: VL-JEPA Research Extension         │
       │  • Joint Embedding Predictive Architecture Integration │
       │  • Latent-space continuous target prediction           │
       │  • Comparative evaluation: Accuracy, Grounding, Speed │
       └────────────────────────────────────────────────────────┘
```

---

# 4. Phase 1 Architecture (Reproduction Baseline)

The reproduced baseline faithfully mirrors the architecture described in:
> Louisa Canepa, Sonit Singh, Arcot Sowmya. **Visual Question Answering in the Medical Domain**. *arXiv:2309.11080*, 2023.

### 4.1 Visual Encoder (VGG-16)
* **Backbone**: VGG-16 pretrained on ImageNet-1k.
* **Input**: $3 \times 224 \times 224$ RGB image, normalized by ImageNet channel mean and standard deviation.
* **Feature Extraction**: Activations extracted from the second fully-connected layer (`classifier[3]`, fc7), producing a 1D feature vector of dimension $d_v = 4096$.
* **Fine-Tuning**: All VGG-16 layers are fine-tuned end-to-end during training.

### 4.2 Question Encoder (BERT)
* **Backbone**: Pretrained `bert-base-uncased` ($12$ layers, $768$ hidden dimensions, $12$ attention heads).
* **Tokenization**: Standard WordPiece tokenizer with maximum sequence length truncated/padded to $32$ tokens.
* **Feature Extraction**: Pooled `[CLS]` token embedding passed through BERT's pooler dense layer with tanh activation, yielding a 1D question vector of dimension $d_q = 768$.
* **Fine-Tuning**: BERT encoder is fine-tuned end-to-end.

### 4.3 Multimodal Fusion & Classification Head
* **Fusion**: Direct feature concatenation:
  $$h_{\text{fused}} = [v \,\|\, q] \in \mathbb{R}^{4096 + 768} = \mathbb{R}^{4864}$$
* **Fully Connected Head**: 2 dense layers with intermediate ReLU and Dropout ($p = 0.3$):
  $$h_1 = \text{Dropout}(\text{ReLU}(W_1 h_{\text{fused}} + b_1)), \quad W_1 \in \mathbb{R}^{1024 \times 4864}$$
  $$\hat{y} = W_2 h_1 + b_2, \quad W_2 \in \mathbb{R}^{C \times 1024}$$
  where $C$ is the number of answer classes ($C \in [1523, 1551]$ depending on the split).
* **Loss Function**: Multi-class Cross-Entropy Loss.

---

# 5. Phase 1 Reproduction Results

The official reproduction was trained across **5 independent 80/20 image-level splits** for 50 epochs each using Adam ($\text{lr} = 10^{-4}$). The results demonstrate a consistent, high-fidelity reproduction within $2.35$ percentage points of the paper's reported mean.

### 5.1 Five-Split Quantitative Summary (Run 4)

| Split | Overall Acc (%) | Strict Acc (%) | Modality (%) | Plane (%) | Organ (%) | Abnormality (%) | Unseen Ceiling (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split 1** (Seed 0) | 57.15 | 57.12 | 66.45 | 78.24 | 72.82 | 11.27 | 92.34 |
| **Split 2** (Seed 1) | 57.99 | 57.96 | 66.62 | 79.10 | 74.52 | 11.21 | 93.43 |
| **Split 3** (Seed 2) | 56.95 | 56.89 | 64.17 | 78.96 | 70.41 | 14.08 | 92.71 |
| **Split 4** (Seed 3) | 58.95 | 58.85 | 67.19 | 78.29 | 74.77 | 14.86 | 94.01 |
| **Split 5** (Seed 4) | 57.20 | 57.11 | 67.32 | 76.31 | 73.05 | 12.26 | 93.07 |
| **Mean** | **57.65** | **57.58** | **66.35** | **78.18** | **73.11** | **12.73** | **93.11** |
| **Sample Std ($s$)** | **± 0.83** | **± 0.82** | **± 1.27** | **± 1.11** | **± 1.74** | **± 1.66** | **± 0.65** |

*(Sample standard deviation computed with $N-1 = 4$ degrees of freedom).*

### 5.2 Direct Comparison with Paper (Table III: BERT + VGG-16)

| Category / Metric | Our Reproduction (Mean ± Std) | Paper Target (Table III) | Gap (Reproduction − Paper) |
| :--- | :---: | :---: | :---: |
| **Modality** | $66.35\% \pm 1.27\%$ | $76.0\%$ | $-9.65\text{ points}$ |
| **Plane** | $78.18\% \pm 1.11\%$ | $77.0\%$ | $+1.18\text{ points}$ |
| **Organ System** | $73.11\% \pm 1.74\%$ | $74.0\%$ | $-0.89\text{ points}$ |
| **Abnormality** | $12.73\% \pm 1.66\%$ | $8.0\%$ | $+4.73\text{ points}$ |
| **Overall (Official Semantics)** | **$57.65\% \pm 0.83\%$** | **$60.0\% \pm 1.0\%$** | **$-2.35\text{ points}$** |
| **Overall (Strict Exact Match)** | $57.58\% \pm 0.82\%$ | — | — |
| **Unseen-Answer Ceiling** | $93.11\% \pm 0.65\%$ | — | — |

### 5.3 Documented Implementation Deviations
To achieve numerical stability and fit hardware constraints without modifying model behavior:
1. **Gradient Clipping (`grad_clip_norm = 5.0`)**: In early unconstrained training, gradient norms periodically spiked above $100.0$, causing catastrophic loss divergence. Clipping at $5.0$ stabilized training completely.
2. **Learning Rate Step Schedule (`lr_step_size = 5, lr_gamma = 0.5`)**: Halving the learning rate every 5 epochs allowed the model to converge smoothly from $10^{-4}$ down to $10^{-6}$.
3. **Micro-Batch Gradient Accumulation (`micro_batch_size = 16`, accumulation $= 4$)**: Allows full reproduction of the paper's effective batch size of $64$ on an 8 GB VRAM GPU without out-of-memory errors.

---

# 6. Dataset — VQA-Med 2019

The project uses the **VQA-Med 2019** benchmark dataset from ImageCLEF 2019.

### 6.1 Verified Dataset Statistics
When all official partitions (Train, Val, Test) are pooled as in Canepa et al., the dataset contains:
* **Total Radiology Images**: Exactly **4,200** unique medical images.
* **Total QA Pairs**: Exactly **15,292** question-answer pairs.
* **Dataset File Checksum**: `data/processed/all_qa_pairs.csv` (MD5: `5cbf955ad6d262b8f2382e8e48fab524`).

### 6.2 Question Category Breakdown
The 15,292 questions are distributed across four clinical categories:

| Category | Total QA Pairs | Description & Example Question |
| :--- | :---: | :--- |
| **Modality** | 3,825 | Imaging technology used (*"Is this a T1 weighted, T2 weighted, or FLAIR image?"*) |
| **Plane** | 3,825 | Anatomical acquisition plane (*"In which plane is the MRI displayed?"*) |
| **Organ System** | 3,825 | Organ or anatomical region shown (*"What is the organ principally shown in this X-ray?"*) |
| **Abnormality** | 3,817 | Clinical pathology or lesion presence (*"What abnormality is seen in the lung apex?"*) |
| **Total** | **15,292** | *(Balanced 25% distribution across the four clinical categories)* |

---

# 7. Dataset Splitting & Split Integrity

### 7.1 Image-Level vs. QA-Level Splitting
In Med-VQA, multiple questions frequently reference the same medical image (on average $\sim 3.64$ questions per image). How the dataset is partitioned is critical:
* **Image-Level Splitting (Our Official Reproduction)**:
  * Partitioning is performed strictly on **unique image IDs**.
  * All QA pairs belonging to an image are assigned exclusively to either Train or Test.
  * **Image overlap between Train and Test = 0%** (zero visual leakage).
  * Exactly $3,360$ images in Train ($80\%$) and $840$ images in Test ($20\%$).
* **QA-Level Splitting (Diagnostic Ablation)**:
  * Partitioning randomly across QA pairs without grouping by image ID.
  * Results in **$94.63\%$ of test images** appearing in the training set with different questions.
  * Used solely as a diagnostic experiment (`experiments/diagnostic_qa_run1/`) to evaluate visual memorization.

### 7.2 Answer Vocabulary & The Unseen-Answer Ceiling
To prevent label leakage, the answer vocabulary is constructed **strictly from the training split rows**:
* **Vocabulary Size**: Between $1,523$ and $1,551$ unique answer classes per split.
* **Unseen Test Answers**: Across the 5 splits, an average of **$6.89\% \pm 0.65\%$** of test rows have answers that never appeared in the training set.
* **Mathematical Ceiling**: Because unseen classes cannot be predicted by a closed-world classifier, the theoretical maximum test accuracy on Image-level splits is bounded at **$93.11\% \pm 0.65\%$**.

---

# 8. Training and Evaluation Strategy

### 8.1 Hyperparameter Specifications
* **Framework**: PyTorch with CUDA acceleration.
* **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{weight\_decay} = 0.0$).
* **Base Learning Rate**: $1 \times 10^{-4}$ with StepLR decay ($\gamma = 0.5$ every 5 epochs).
* **Batch Size**: Effective batch size $64$ (realized via micro-batch $16 \times 4$ gradient accumulation steps).
* **Epochs**: $50$ epochs per split.
* **Seeds**: Seeds $0, 1, 2, 3, 4$ for Splits $1, 2, 3, 4, 5$.

### 8.2 Image Augmentation Pipeline (Training Only)
Implemented via `src/preprocessing.py` and torchvision transforms:
* **Color Jitter**: Brightness $\pm 5\%$, Contrast $\pm 5\%$ ($p = 0.4$).
* **Affine Transform**: Translation $\pm 5\text{ px}$, Rotation $\pm 5^\circ$ ($p = 0.5$).
* **Gaussian Blur**: Kernel size $5 \times 5$, $\sigma \in [0.1, 2.0]$ ($p = 0.5$).
* **Gaussian Noise**: $\sigma = 0.01$ additive zero-mean Gaussian noise ($p = 0.4$).

### 8.3 Evaluation Semantics
* **Official Semantics**: Matches predictions against the ground-truth answer accounting for dataset `#` delimiters (e.g., `A#B` indicates acceptable alternative synonyms) and optional parenthesized expansions (e.g., `(left) lung` accepts both `left lung` and `lung`).
* **Strict Exact Match**: Requires exact character-for-character match with the first literal string.

---

# 9. Explainability & Evidence Verification

To audit what visual features drive model decisions, the project includes:
1. **Grad-CAM Visual Explanations** (`src/explainability/gradcam.py`):
   * Computes gradients with respect to VGG-16's final convolutional layer (`features[28]`, $512 \times 14 \times 14$).
   * Generates overlay heatmaps showing image regions contributing most to the predicted answer class.
2. **Shortcut Border Energy Audits** (`scripts/analyze_shortcuts.py`):
   * Quantifies whether Grad-CAM energy concentrates in clinically relevant central lung/brain parenchyma vs. non-anatomical scan borders, orientation markers, or corner labels.

---

# 10. Phase 2 — JEPA Research Extension

Having verified the baseline reproduction and identified its reliance on closed-world classification, Phase 2 explores **Joint Embedding Predictive Architecture (JEPA)** for Med-VQA.

### 10.1 Motivation for Vision-Language JEPA in Med-VQA
Standard Med-VQA models treat answering as an $N$-way classification task over discrete label indices. This causes several fundamental limitations:
1. **Zero Generalization to Unseen Clinical Findings**: Any answer not seen during training cannot be predicted.
2. **Loss of Semantic Proximity**: Cross-entropy penalizes a clinically close synonym (e.g., predicting *"pneumonia"* instead of *"consolidation"*) identically to an absurd answer (e.g., *"ct scan"*).
3. **Susceptibility to Shortcut Learning**: The model maps visual artifacts directly to high-frequency class logits without learning continuous multimodal representations.

### 10.2 The VL-JEPA Approach
Based on Chen et al. (*VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language*, 2025/2026), JEPA models:
* Predict representations in an **abstract embedding space** rather than computing categorical softmax logits or autoregressively generating text tokens.
* Train a predictor network to map `[Image, Question]` representations directly into the continuous latent representation of the clinical answer:
  $$\mathcal{L}_{\text{JEPA}} = \mathcal{D}(s_\phi(E_V(I), E_Q(Q)), \, E_A(A))$$
* Evaluate answers via nearest-neighbor embedding retrieval or linear probing in the latent space.

---

# 11. Project Structure

```
MedVQA/
├── data/
│   ├── raw/                                 # Raw downloaded ImageCLEF 2019 files
│   ├── processed/
│   │   └── all_qa_pairs.csv                 # Canonical 15,292 QA pairs (MD5: 5cbf955a...)
│   └── splits/
│       └── image/                           # 5 official image-level splits (80/20)
│           ├── split_1.json ... split_5.json
│           └── summary.json
│
├── src/
│   ├── config.py                            # Central frozen configuration
│   ├── dataset.py                           # PyTorch Dataset, caching, tokenization
│   ├── preprocessing.py                     # Image augmentations & normalization
│   ├── answers.py                           # Answer vocabulary & semantic matching
│   ├── models/
│   │   ├── image_encoder.py                 # VGG-16 backbone (4096-d fc7)
│   │   ├── question_encoder.py              # BERT-base backbone (768-d pooler)
│   │   ├── fusion.py                        # Multimodal concatenation
│   │   └── medvqa_model.py                  # End-to-end classifier
│   ├── training/
│   │   ├── trainer.py                       # 50-epoch loop with accumulation & clipping
│   │   ├── checkpoint.py                    # Atomic checkpointing & recovery
│   │   └── losses.py                        # Cross-entropy criterion
│   ├── evaluation/
│   │   ├── evaluate.py                      # Evaluation runner (official & strict)
│   │   └── metrics.py                       # Accuracy & category metrics
│   └── explainability/
│       └── gradcam.py                       # Grad-CAM on VGG-16 features.28
│
├── checkpoints/
│   └── run4/                                # Frozen official checkpoints (splits 1..5)
│       └── split_1/ ... split_5/            # last.pt, best.pt, history.json, train.log
│
├── results/
│   ├── reproduction/
│   │   └── run4/                            # Official reproduction plots & logs
│   │       ├── split_1/ ... split_5/        # overview.png, summary.json
│   │       └── shortcut_analysis_5splits_*  # 5-split shortcut metrics & plots
│   └── gradcam/                             # Grad-CAM visual overlays
│
├── experiments/
│   └── diagnostic_qa_run1/                  # QA-level split diagnostic ablation
│       ├── split/split_1.json               # 80/20 QA-level split (94.6% image overlap)
│       ├── train_diagnostic.py              # Self-contained diagnostic runner
│       ├── checkpoint/                      # Isolated diagnostic checkpoints
│       └── results/                         # Live overview plots & metrics
│
├── backups/
│   └── run4_frozen/                         # Complete immutable backup of run4
│
├── scripts/
│   ├── summarize_run.py                     # 5-split summary table generator
│   ├── plot_run.py                          # 4-panel training dynamics plotter
│   ├── run_gradcam.py                       # Grad-CAM visualization runner
│   ├── analyze_shortcuts_5splits.py         # Multi-split shortcut energy analyzer
│   └── split_dataset.py                     # Dataset split generator
│
├── train.py                                 # Main reproduction launcher
├── requirements.txt                         # Python dependencies
└── README.md                                # Main project documentation
```

---

# 12. Development Roadmap

### Phase 1: Baseline Reproduction (Completed)
- [x] Environment verification (PyTorch 2.14, CUDA, RTX 5060 Laptop GPU).
- [x] Dataset consolidation & MD5 verification ($15,292$ QA pairs, $4,200$ images).
- [x] 5-split image-level partitioning ($0\%$ visual leakage, $80/20$ ratio).
- [x] VGG-16 ($4096$-d) + BERT-base ($768$-d) multimodal architecture implementation.
- [x] Gradient clipping ($5.0$) and StepLR decay schedule stabilization.
- [x] Training of all 5 reproduction splits ($50$ epochs each).
- [x] Final quantitative evaluation ($57.65\% \pm 0.83\%$ vs Paper $60.0\% \pm 1.0\%$).
- [x] Explainability analysis using Grad-CAM and border shortcut energy audits.
- [x] Diagnostic QA-split experiment creation and verification.
- [x] Permanent backup and freeze of all reproduction checkpoints and artifacts.

### Phase 2: JEPA Research Extension (Upcoming)
- [ ] Implement Vision-Language JEPA encoder and target predictor.
- [ ] Adapt continuous text embedding target representations for medical answers.
- [ ] Train JEPA architecture on VQA-Med 2019 under identical 5 image-level splits.
- [ ] Compare JEPA vs. Baseline on accuracy, rare abnormality retrieval, and visual grounding.
- [ ] Conduct final comparative analysis and publish Honors thesis report.

---

# 13. Hardware & Compute Environment

All experiments in this repository are executed on a native Windows workstation:
* **OS**: Windows 11
* **CPU**: AMD Ryzen 9 8900HX
* **RAM**: 16 GB DDR5
* **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6 VRAM)
* **Software**: Python 3.11/3.12 (Conda `medvqa`), PyTorch 2.14.1 with CUDA 13.0, HuggingFace Transformers, Torchvision.

---

# 14. References & Citations

1. **VQA-Med 2019 Dataset**:
   > Asma Ben Abacha, Sadid A. Hasan, Vivek V. Datla, Joey Liu, Dina Demner-Fushman, Henning Müller. *VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019*. Working Notes of CLEF, 2019. [ceur-ws.org/Vol-2380/paper_272.pdf](https://ceur-ws.org/Vol-2380/paper_272.pdf).

2. **Baseline Paper (Reproduction Target)**:
   > Louisa Canepa, Sonit Singh, Arcot Sowmya. *Visual Question Answering in the Medical Domain*. arXiv:2309.11080 [cs.CV], 2023. [doi.org/10.48550/arXiv.2309.11080](https://doi.org/10.48550/arXiv.2309.11080).

3. **VL-JEPA (Extension Target)**:
   > Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung. *VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language*. arXiv:2512.10942 [cs.CV], 2025/2026. [arxiv.org/abs/2512.10942](https://arxiv.org/abs/2512.10942).

```bibtex
@article{canepa2023visual,
  title={Visual Question Answering in the Medical Domain},
  author={Canepa, Louisa and Singh, Sonit and Sowmya, Arcot},
  journal={arXiv preprint arXiv:2309.11080},
  year={2023}
}

@inproceedings{ImageCLEFVQA-Med2019,
  title={VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019},
  author={Ben Abacha, Asma and Hasan, Sadid A. and Datla, Vivek V. and Liu, Joey and Demner-Fushman, Dina and M{\"u}ller, Henning},
  booktitle={Working Notes of CLEF 2019},
  year={2019}
}

@misc{chen2026vljepa,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language},
  author={Chen, Delong and Shukor, Mustafa and Moutakanni, Theo and Chung, Willy and Yu, Jade and Kasarla, Tejaswi and Bang, Yejin and Bolourchi, Allen and LeCun, Yann and Fung, Pascale},
  year={2026},
  eprint={2512.10942},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
