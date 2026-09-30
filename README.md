# Honors-MedVQA-Reproduction

## Medical Visual Question Answering with VGG-16, BERT, and JEPA

This repository contains the implementation and experimental reproduction for an Honors research project on **Medical Visual Question Answering (Med-VQA)**.

The project is divided into two major research phases:

1. **Reproduction of the Med-VQA architecture proposed by Canepa et al.**
2. **Integration and investigation of Joint Embedding Predictive Architecture (JEPA) / Vision-Language JEPA into the reproduced Med-VQA system**

The objective is to first establish a faithful and reproducible baseline from the published work and then investigate how a JEPA-based vision-language representation can be incorporated into the medical VQA setting.

---

# 1. Project Motivation

Medical Visual Question Answering is a multimodal machine learning task in which a model receives a medical image and a natural-language question and predicts an answer.

The task can be represented as:

```text
                 Medical Image
                      +
                    Question
                      |
                      v
                Med-VQA Model
                      |
                      v
                    Answer
```

Medical VQA is particularly challenging because the model must understand both:

* visual information contained in medical images
* semantic information contained in natural-language questions

The VQA-Med 2019 dataset provides a benchmark for this problem using radiology images and clinically oriented questions.

This project investigates a relatively compact multimodal architecture first and then explores a JEPA-based extension.

---

# 2. Research Questions

The project is organized around the following research questions.

### RQ1 — Reproduction

Can the architecture described by Canepa et al. be reproduced using the VQA-Med 2019 dataset and obtain performance comparable to the reported results?

### RQ2 — JEPA Integration

Can the JEPA / VL-JEPA representation-learning approach be incorporated into the reproduced Med-VQA architecture?

### RQ3 — Comparison

How does the JEPA-based extension compare with the reproduced baseline in terms of:

* accuracy
* training behavior
* parameterization
* representation quality
* visual evidence
* computational requirements

### RQ4 — Evidence Verification

Can visual explanation methods such as GradCAM provide evidence about which image regions contribute to the model's predictions?

---

# 3. Overall Research Pipeline

The complete project follows this progression:

```text
                    VQA-Med 2019
                         |
                         v
                 Dataset Analysis
                         |
                         v
              Paper Architecture
                         |
                         v
             VGG-16 + BERT + Fusion
                         |
                         v
                 Reproduction
                         |
                         v
                Baseline Results
                         |
                         v
                    GradCAM
                         |
                         |
                         v
              ┌────────────────────┐
              │   JEPA / VL-JEPA   │
              │      Extension     │
              └─────────┬──────────┘
                        |
                        v
                JEPA-Based Model
                        |
                        v
                  Evaluation
                        |
                        v
                 Comparison
```

The baseline reproduction is intentionally completed **before** introducing JEPA.

This allows the effect of the JEPA modification to be studied relative to a known baseline.

---

# 4. Phase 1 — Reproduction of the Published Med-VQA Architecture

The first phase focuses exclusively on reproducing the architecture from:

> Louisa Canepa, Sonit Singh, Arcot Sowmya
> **Visual Question Answering in the Medical Domain**

The paper investigates approaches for Med-VQA and reports a model achieving approximately 60% accuracy on the VQA-Med 2019 test set.

Paper:

https://arxiv.org/abs/2309.11080

The architecture selected for reproduction is the configuration using:

```text
VGG-16
+
BERT
+
Concatenation
+
Fully Connected Layers
```

---

# 5. Phase 1 Architecture

The reproduced architecture contains two main encoders.

## 5.1 Visual Encoder

The medical image is processed using VGG-16.

```text
Medical Image
      |
      v
   VGG-16
      |
      v
Image Feature Representation
```

The VGG-16 network provides the visual representation used by the multimodal model.

---

## 5.2 Question Encoder

The natural-language question is processed using BERT.

```text
Question
   |
   v
Tokenizer
   |
   v
BERT
   |
   v
Question Feature Representation
```

The BERT representation provides the semantic representation of the question.

---

## 5.3 Multimodal Fusion

The visual and textual representations are combined using concatenation.

```text
                 Image Features
                       |
                       |
                       v
                  +---------+
                  |         |
                  | CONCAT  |
                  |         |
                  +---------+
                       ^
                       |
                       |
                Question Features
```

The resulting multimodal representation is then passed through fully connected layers.

```text
Concatenated Representation
            |
            v
     Fully Connected
            |
            v
     Fully Connected
            |
            v
       Answer Class
```

---

# 6. Phase 1 Target

The first experimental target is to reproduce the reported performance of the architecture as closely as possible.

The reported target is approximately:

```text
Accuracy ≈ 60%
```

The exact reproduction result obtained by this repository will be reported separately.

We will not artificially modify the model to force a particular accuracy.

Instead:

```text
Paper Configuration
       |
       v
Our Implementation
       |
       v
Our Experimental Result
       |
       v
Comparison with Paper
```

Any implementation detail not explicitly specified in the paper will be documented as an implementation choice.

---

# 7. Dataset — VQA-Med 2019

The project uses the **VQA-Med 2019** dataset introduced for the ImageCLEF 2019 Medical Visual Question Answering task.

The dataset focuses on radiology images and four major question categories:

```text
1. Modality
2. Plane
3. Organ System
4. Abnormality
```

The original VQA-Med 2019 repository provides the training, validation, and test data.

Dataset repository:

https://github.com/abachaa/VQA-Med-2019

The dataset documentation specifies:

```text
Training:
3,200 images
12,792 QA pairs

Validation:
500 images
2,000 QA pairs

Test:
500 images
500 questions
```

The dataset therefore contains:

```text
4,200 images
15,292 questions / QA pairs
```

The official dataset paper also reports 4,200 radiology images and 15,292 question-answer pairs.

---

# 8. Dataset Categories

## Modality

Questions concerning the imaging modality.

Examples include:

```text
Is this a T1 weighted, T2 weighted, or FLAIR image?
```

## Plane

Questions concerning the imaging plane.

Examples include:

```text
In which plane is the MRI displayed?
```

## Organ System

Questions concerning the primary organ/system shown in the image.

Example:

```text
What is the organ principally shown in this X-ray?
```

## Abnormality

Questions concerning abnormalities visible in the image.

Example:

```text
What is most alarming about this CT scan?
```

---

# 9. Dataset Organization

The original downloaded dataset is kept under:

```text
data/raw/
```

Processed data generated by the project is stored separately.

```text
data/
│
├── raw/
│   ├── images/
│   └── annotations/
│
└── processed/
    ├── train/
    ├── test/
    └── answer_vocab/
```

The original dataset is not modified directly.

---

# 10. Dataset Reproducibility

The original dataset files are not committed to this Git repository.

Reasons:

* dataset size
* repository size
* data distribution requirements
* reproducibility

The dataset must therefore be downloaded separately from the official VQA-Med 2019 source.

The original VQA-Med repository also provides the dataset documentation and citation information.

---

# 11. Important Dataset Count Note

The VQA-Med 2019 dataset documentation lists:

```text
Training: 3,200 images / 12,792 QA pairs
Validation: 500 images / 2,000 QA pairs
Test: 500 images / 500 questions
```

The original dataset paper reports:

```text
4,200 radiology images
15,292 question-answer pairs
```

These counts are consistent when the official splits are considered.

Any additional counts reported by later papers will be treated as paper-specific descriptions or experimental subsets and will be verified against the actual files used by the reproduction.

---

# 12. Training and Evaluation Strategy

The implementation will follow the experimental setup described by the paper wherever the paper provides sufficient information.

Important reported settings include:

```text
Framework:
PyTorch

Optimizer:
Adam

Learning Rate:
1e-4

Batch Size:
64

Training:
50 epochs

Evaluation:
5-fold cross-validation

Train/Test:
80/20
```

The exact implementation will be recorded in the project configuration.

---

# 13. Image Preprocessing

The reproduction will implement the image preprocessing and augmentation described by the paper.

The reported augmentation operations include:

```text
Brightness / contrast adjustment
Translation / rotation
Gaussian blur
Gaussian noise
```

The preprocessing pipeline will be implemented separately from the model.

```text
Raw Image
    |
    v
Image Preprocessing
    |
    v
Normalized / Augmented Image
    |
    v
VGG-16
```

---

# 14. Question Processing

The question pipeline is:

```text
Raw Question
     |
     v
BERT Tokenizer
     |
     v
Token IDs / Attention Mask
     |
     v
BERT
     |
     v
Question Representation
```

The exact tokenizer and BERT configuration used in the implementation will be recorded in the project configuration.

---

# 15. Answer Prediction

The task is implemented as answer classification.

The overall Phase 1 pipeline is:

```text
                     Medical Image
                          |
                          v
                       VGG-16
                          |
                          v
                   Image Features
                          |
                          |
                          +----------------+
                                           |
                                           v
                                      Concatenate
                                           ^
                                           |
                                           +----------------+
                                           |
                                    Question Features
                                           ^
                                           |
                                         BERT
                                           ^
                                           |
                                        Question
                                           |
                                           v
                                  Fully Connected
                                           |
                                           v
                                  Answer Prediction
```

---

# 16. Checkpointing and Crash Recovery

Training is designed to be recoverable.

Because the experiment may involve multiple epochs and cross-validation folds, the model will periodically save checkpoints.

Checkpoint structure:

```text
checkpoints/
│
├── fold_1/
│   ├── latest.pt
│   └── best.pt
│
├── fold_2/
│   ├── latest.pt
│   └── best.pt
│
├── fold_3/
│   ├── latest.pt
│   └── best.pt
│
├── fold_4/
│   ├── latest.pt
│   └── best.pt
│
└── fold_5/
    ├── latest.pt
    └── best.pt
```

A checkpoint may contain:

```text
Model state
Optimizer state
Epoch
Fold
Best metric
Training metrics
Validation metrics
Random state information
```

The intended workflow is:

```text
Train
  |
  v
Save checkpoint
  |
  X
Crash / interruption
  |
  v
Restart
  |
  v
Load checkpoint
  |
  v
Resume training
```

This prevents unnecessary retraining from epoch 1 after an interruption.

---

# 17. Evaluation

The primary evaluation metric is:

```text
Accuracy
```

Where applicable, results will also be analyzed by question category:

```text
Modality
Plane
Organ System
Abnormality
```

Evaluation outputs will be stored under:

```text
results/
├── metrics/
├── predictions/
└── gradcam/
```

---

# 18. GradCAM and Evidence Verification

The project will also investigate visual evidence using GradCAM.

The purpose is to inspect which regions of the medical image contribute to the model's prediction.

Pipeline:

```text
Image + Question
       |
       v
    Med-VQA
       |
       v
Prediction
       |
       v
   GradCAM
       |
       v
Heatmap
       |
       v
Medical Image with Visual Evidence
```

GradCAM results will be stored in:

```text
results/gradcam/
```

This allows qualitative analysis of model behavior in addition to numerical accuracy.

---

# 19. Phase 2 — JEPA Extension

After completing Phase 1, the project will investigate the use of **Joint Embedding Predictive Architecture (JEPA)** for the Med-VQA task.

The main JEPA reference is:

> Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung
> **VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language**

arXiv:

https://arxiv.org/abs/2512.10942

The paper describes a vision-language architecture that predicts continuous representations of target text rather than generating text token-by-token autoregressively. It also evaluates the approach on discriminative VQA tasks.

---

# 20. Why VL-JEPA Is Relevant to This Project

Traditional multimodal VQA systems commonly map:

```text
Image + Question
       |
       v
Multimodal Representation
       |
       v
Answer
```

The JEPA approach instead emphasizes prediction in an abstract embedding space.

Conceptually:

```text
Image + Question
       |
       v
Vision-Language Encoder
       |
       v
Predictor
       |
       v
Predicted Text Embedding
```

The VL-JEPA paper describes this as predicting continuous embeddings of target text rather than autoregressively generating individual tokens.

This project will investigate whether this representation-learning approach can be adapted to the medical VQA setting.

---

# 21. Phase 2 Experimental Direction

The JEPA experiment will be built **after** the Phase 1 reproduction is established.

The baseline remains:

```text
VGG-16
   +
BERT
   +
Concatenation
   +
FC Layers
   =
Baseline Med-VQA
```

The second phase will investigate a JEPA-based alternative.

Conceptually:

```text
                 PHASE 1
                    |
                    v
        VGG-16 + BERT + Concatenation
                    |
                    v
              Baseline Model
                    |
                    v
               Evaluation
                    |
                    |
                    v
                 PHASE 2
                    |
                    v
            JEPA Integration
                    |
                    v
            JEPA-based Model
                    |
                    v
               Evaluation
                    |
                    v
              Comparison
```

The exact integration point will be determined during implementation based on the requirements and architecture described in the VL-JEPA paper.

---

# 22. Baseline vs JEPA Experiment

The experiments will be kept separate.

## Experiment A — Paper Reproduction

```text
VQA-Med 2019
      |
      v
VGG-16
      +
BERT
      |
      v
Concatenation
      |
      v
FC Layers
      |
      v
Answer
```

## Experiment B — JEPA Extension

```text
VQA-Med 2019
      |
      v
JEPA-based Vision-Language Representation
      |
      v
Medical VQA Prediction
      |
      v
Answer
```

The two experiments will use comparable evaluation procedures wherever possible.

---

# 23. What Will Be Compared

The final experimental analysis will compare the baseline and JEPA-based system using:

### Performance

```text
Overall Accuracy
Category-level Accuracy
```

### Training

```text
Training Loss
Validation / Evaluation Performance
Convergence Behavior
```

### Computational Characteristics

```text
Trainable Parameters
GPU Memory Usage
Training Time
Inference Time
```

### Explainability

```text
GradCAM
Visual Evidence
Prediction Examples
```

---

# 24. Research Reproducibility Principle

A central principle of this project is:

> **First reproduce, then modify.**

The baseline implementation should remain as close as possible to the published architecture.

Only after the baseline has been evaluated will modifications be introduced.

The project will distinguish between:

```text
Paper-specified behavior
```

and

```text
Implementation decisions required to reproduce the experiment
```

This prevents undocumented modifications from being presented as part of the original paper.

---

# 25. Project Structure

```text
MedVqa/
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   │
│   └── processed/
│       ├── train/
│       ├── test/
│       └── answer_vocab/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_encoder.py
│   │   ├── question_encoder.py
│   │   ├── fusion.py
│   │   └── medvqa_model.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── checkpoint.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   └── explainability/
│       ├── __init__.py
│       └── gradcam.py
│
├── checkpoints/
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
│
├── experiments/
│   └── reproduction/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── gradcam/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_vgg16_test.ipynb
│   ├── 03_bert_test.ipynb
│   └── 04_model_test.ipynb
│
├── scripts/
│   ├── download_data.py
│   └── test_environment.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── train.py
```

---

# 26. Development Roadmap

## Stage 1 — Environment

* [ ] Verify Python environment
* [ ] Verify PyTorch installation
* [ ] Verify CUDA
* [ ] Verify RTX 5060
* [ ] Verify GPU computation
* [ ] Record package versions

## Stage 2 — Dataset

* [ ] Obtain VQA-Med 2019
* [ ] Place raw dataset under `data/raw/`
* [ ] Inspect annotation files
* [ ] Verify image counts
* [ ] Verify QA counts
* [ ] Verify answer categories
* [ ] Understand train/validation/test structure
* [ ] Implement dataset loader

## Stage 3 — Preprocessing

* [ ] Image normalization
* [ ] Image augmentation
* [ ] Question tokenization
* [ ] Answer vocabulary
* [ ] Label encoding

## Stage 4 — Visual Encoder

* [ ] Load VGG-16
* [ ] Verify pretrained weights
* [ ] Test image forward pass
* [ ] Verify feature dimensions

## Stage 5 — Text Encoder

* [ ] Load BERT
* [ ] Verify tokenizer
* [ ] Test question encoding
* [ ] Verify feature dimensions

## Stage 6 — Baseline Architecture

* [ ] Implement feature concatenation
* [ ] Implement fully connected layers
* [ ] Implement answer classifier
* [ ] Verify complete forward pass
* [ ] Verify backward pass

## Stage 7 — Training

* [ ] Implement Adam optimizer
* [ ] Implement training loop
* [ ] Implement evaluation loop
* [ ] Implement checkpointing
* [ ] Implement resume functionality
* [ ] Test on a small subset
* [ ] Run full experiment

## Stage 8 — Reproduction

* [ ] Run required folds
* [ ] Record metrics
* [ ] Compare against reported paper results
* [ ] Save predictions
* [ ] Save experiment configuration

## Stage 9 — Explainability

* [ ] Implement GradCAM
* [ ] Generate visual explanations
* [ ] Analyze successful predictions
* [ ] Analyze incorrect predictions

## Stage 10 — JEPA

* [ ] Study VL-JEPA architecture
* [ ] Identify suitable integration point
* [ ] Implement JEPA component
* [ ] Create separate JEPA experiment
* [ ] Train with checkpointing
* [ ] Evaluate
* [ ] Compare with baseline

## Stage 11 — Final Analysis

* [ ] Accuracy comparison
* [ ] Category-wise comparison
* [ ] Training behavior
* [ ] Computational comparison
* [ ] GradCAM comparison
* [ ] Error analysis
* [ ] Final research conclusions

---

# 27. Environment

The project is developed and tested using a native Windows environment.

Primary hardware:

```text
CPU:
AMD Ryzen 9 8900HX

RAM:
16 GB

GPU:
NVIDIA GeForce RTX 5060 Laptop GPU

VRAM:
8 GB
```

The primary deep-learning framework is:

```text
PyTorch
```

CUDA acceleration is used for model training and inference.

WSL is not required for the primary implementation.

---

# 28. Reproducibility and Version Control

Git is used to track the source code.

Large datasets and model checkpoints are intentionally excluded from Git.

Excluded directories include:

```text
data/raw/
data/processed/
checkpoints/
results/
.venv/
```

The repository tracks:

```text
Source code
Configuration
Notebooks
Scripts
Documentation
Requirements
Experiment definitions
```

---

# 29. Expected Final Outcome

The project is expected to produce two major experimental systems.

### System 1 — Reproduced Paper Architecture

```text
VGG-16
+
BERT
+
Concatenation
+
Fully Connected Layers
```

This provides the reproduction baseline.

### System 2 — JEPA-Based Extension

```text
Vision-Language Representation
+
JEPA
+
Medical VQA
```

This provides the research extension.

The final analysis will compare the two systems using quantitative and qualitative evidence.

---

# 30. References

## [1] VQA-Med 2019 Dataset

A. Ben Abacha, S. A. Hasan, V. V. Datla, J. Liu, D. Demner-Fushman, and H. Müller.

**VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019.**

Working Notes of CLEF 2019.

Dataset:

https://github.com/abachaa/VQA-Med-2019

Paper:

https://ceur-ws.org/Vol-2380/paper_272.pdf

The original dataset paper describes the ImageCLEF 2019 VQA-Med task and its four question categories: Modality, Plane, Organ System, and Abnormality.

---

## [2] Med-VQA Architecture Reproduction

L. Canepa, S. Singh, and A. Sowmya.

**Visual Question Answering in the Medical Domain.**

arXiv:2309.11080 [cs.CV].

2023.

Paper:

https://arxiv.org/abs/2309.11080

DOI:

https://doi.org/10.48550/arXiv.2309.11080

This is the primary paper reproduced in Phase 1 of this project.

---

## [3] JEPA / Vision-Language JEPA

D. Chen, M. Shukor, T. Moutakanni, W. Chung, J. Yu, T. Kasarla, Y. Bang, A. Bolourchi, Y. LeCun, and P. Fung.

**VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.**

arXiv:2512.10942 [cs.CV].

2025/2026.

Paper:

https://arxiv.org/abs/2512.10942

The paper presents a vision-language JEPA architecture based on predicting continuous text embeddings rather than autoregressive token generation and evaluates its use for vision-language tasks including discriminative VQA.

---

# 31. Citation

If this repository or its implementation is used in future work, please cite the relevant original works above.

### VQA-Med 2019

```bibtex
@Inproceedings{ImageCLEFVQA-Med2019,
  author = {Asma {Ben Abacha} and Sadid A. Hasan and Vivek V. Datla
            and Joey Liu and Dina Demner-Fushman and Henning Müller},
  title = {VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019},
  booktitle = {Working Notes of CLEF 2019},
  series = {CEUR Workshop Proceedings},
  volume = {2380},
  year = {2019},
  publisher = {CEUR-WS.org},
  month = {September 9-12},
  address = {Lugano, Switzerland}
}
```

### Visual Question Answering in the Medical Domain

```bibtex
@article{canepa2023visual,
  title={Visual Question Answering in the Medical Domain},
  author={Canepa, Louisa and Singh, Sonit and Sowmya, Arcot},
  journal={arXiv preprint arXiv:2309.11080},
  year={2023}
}
```

### VL-JEPA

```bibtex
@misc{chen2026vljepa,
  title={VL-JEPA: Joint Embedding Predictive Architecture for Vision-language},
  author={Delong Chen and Mustafa Shukor and Theo Moutakanni
          and Willy Chung and Jade Yu and Tejaswi Kasarla
          and Yejin Bang and Allen Bolourchi and Yann LeCun
          and Pascale Fung},
  year={2026},
  eprint={2512.10942},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.10942}
}
```

---

# 32. Project Status

```text
Project:
Honors Med-VQA Reproduction

Phase 1:
Paper reproduction

Status:
In progress

Baseline:
VGG-16 + BERT + Concatenation

Dataset:
VQA-Med 2019

Phase 2:
JEPA / VL-JEPA extension

Status:
Planned after baseline reproduction

Explainability:
GradCAM

Training:
GPU accelerated with PyTorch + CUDA

Checkpointing:
Enabled for crash recovery
```

---

# 33. Core Principle

```text
              REPRODUCE
                  ↓
              VALIDATE
                  ↓
             UNDERSTAND
                  ↓
               EXTEND
                  ↓
               COMPARE
                  ↓
              ANALYZE
```

The project does not begin by modifying the published architecture.

The published architecture is first implemented and evaluated as the baseline.

JEPA is then introduced as a separate research extension so that its effect can be evaluated against the reproduced baseline.
