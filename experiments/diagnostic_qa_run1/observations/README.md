# QA Split Diagnostic Run 1: Methodology, Setup, and Protocol

## 1. Purpose of the Diagnostic Experiment
The purpose of this experiment is to isolate and evaluate the impact of **split granularity** on Med-VQA model performance. Specifically, it investigates whether partitioning the dataset at the **QA-pair level** rather than the **image level** materially affects test accuracy across question categories.

## 2. Why QA-Level Splitting Was Tested
In our official 5-split reproduction, the overall test accuracy achieved was **57.65% ± 0.83%** (against Canepa et al.'s reported **60.0% ± 1.0%**). The largest single discrepancy occurred in the **Modality** category, which lagged by **-9.65 percentage points** (66.35% reproduced vs. 76.0% paper). 

Canepa et al. stated only that an 80/20 train/test split was performed, but did not specify whether the split unit was the **image** or the **QA pair**. If questions regarding the same image are partitioned across both train and test sets, the model may encounter images during testing that it already observed during training. This diagnostic run tests the performance implications of that protocol.

## 3. Difference Between Image-Level and QA-Level Splitting
* **Image-Level Splitting (Official Reproduction):**
  * All QA pairs associated with a given `image_id` are grouped strictly into either the training set or the test set.
  * Zero image overlap between train and test: $\text{Train Images} \cap \text{Test Images} = \emptyset$.
* **QA-Level Splitting (Diagnostic Run 1):**
  * Each question-answer row is partitioned independently at random (80% train, 20% test).
  * Multiple questions for the same image can be distributed across both partitions.
  * Substantial image overlap between train and test is **expected and intentional**.

## 4. Expected Train/Test Image Overlap in QA Splitting
In Med-VQA 2019, each image typically has between 1 and 4 associated questions (often one per category: modality, plane, organ, abnormality). When splitting 15,292 QA pairs uniformly at random:
* Test QA pairs: 3,058
* Test unique images: 2,292
* **Shared images appearing in BOTH train and test:** **2,169 images** (**94.6%** of all test images).

## 5. Scope and Boundaries
* **NOT Part of the Official Reproduction:** This experiment is strictly a diagnostic ablation.
* **Paper Ambiguity:** The paper does not explicitly document the split unit; therefore, we do NOT claim the paper used QA-level splitting, nor that this experiment represents the paper's actual methodology.

## 6. What Changed vs. What Did NOT Change
* **What Changed (The Sole Independent Variable):**
  * `split_unit`: `"image"` $\rightarrow$ `"qa"` (using seed 0).
* **What Did NOT Change (Frozen Controls):**
  * Model architecture: VGG-16 + BERT-base-uncased + concatenation fusion + 2 FC layers.
  * Dataset: Exact same 15,292 rows in `data/processed/all_qa_pairs.csv` (MD5: `5cbf955ad6d262b8f2382e8e48fab524`).
  * Vocabulary construction: Built **strictly from training rows only** (1,523 classes in image split vs. 1,523 in QA split).
  * Training parameters: 50 epochs, effective batch 64 (micro-batch 16, accum 4), Adam optimizer, initial lr 1e-4 with 0.5x decay every 5 epochs, grad clip 5.0.
  * Evaluation semantics: Exact match against literal and alternate/parenthesized reference forms.

## 7. Dataset and Split Statistics
* Total QA pairs: 15,292
* Total unique images: 4,200
* Training QA pairs: 12,234 (80.0%)
* Testing QA pairs: 3,058 (20.0%)
* Training unique images: 4,077
* Testing unique images: 2,292
* **Images in both Train and Test:** 2,169 (94.6% of test images)
* Unseen test answers in training vocabulary: 208 (6.80%)

## 8. Directory Isolation
All artifacts, logs, checkpoints, and outputs for this experiment are strictly confined to:
`experiments/diagnostic_qa_run1/`
* Split: `experiments/diagnostic_qa_run1/split/split_1.json`
* Checkpoints: `experiments/diagnostic_qa_run1/checkpoint/`
* Results: `experiments/diagnostic_qa_run1/results/`
* Logs: `experiments/diagnostic_qa_run1/logs/`
* Observations: `experiments/diagnostic_qa_run1/observations/`

## 9. Baseline Comparison Reference (Image Run 1)
Upon manual execution, results will be benchmarked against Split 1 of the official image reproduction:
* Overall Accuracy: 57.15% (Strict: 57.12%)
* Modality Accuracy: 66.45%
* Plane Accuracy: 78.24%
* Organ Accuracy: 72.82%
* Abnormality Accuracy: 11.27%

## 10. Scientific Limitations
1. **Split Behavior vs. Patient Leakage:** Demonstrating that QA splitting allows image overlap tests split granularity only; it does **not** establish patient- or case-level leakage across distinct images.
2. **Causal Interpretation:** Higher accuracy under QA splitting indicates that image familiarity aids prediction; it does not prove cognitive medical comprehension.
