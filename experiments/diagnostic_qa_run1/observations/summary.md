# QA Split Diagnostic Run 1: Executive Summary

## Purpose
This diagnostic experiment investigates whether partitioning the Med-VQA 2019 dataset at the **QA-pair level** rather than the **image level** materially affects test accuracy across question categories. It was designed to assess whether the -9.65 point discrepancy in Modality accuracy between our official image-level reproduction (66.35%) and the paper's Table III (76.0%) could be driven by the choice of split granularity.

## Split Protocol
* **Split Type:** QA-pair level random partition (`split_unit = "qa"`).
* **Ratio:** 80% train (12,234 QA pairs) / 20% test (3,058 QA pairs).
* **Seed:** 0 (`np.random.default_rng(0)`).
* **Image Overlap:** The same `image_id` is permitted in both train and test. Of the 2,292 unique test images, **2,169 images (94.6%)** also appear in the training partition with different questions.
* **Integrity:** `train_idx` and `test_idx` are strictly disjoint; all 15,292 rows are accounted for; CSV MD5 matches `5cbf955ad6d262b8f2382e8e48fab524`.

## Results
*(Training pending manual execution by user)*

| Metric | Image Run 1 (Split 1) | QA Diagnostic Run 1 | Difference |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 57.15% | *(Pending manual run)* | -- |
| **Modality Accuracy** | 66.45% | *(Pending manual run)* | -- |
| **Plane Accuracy** | 78.24% | *(Pending manual run)* | -- |
| **Organ Accuracy** | 72.82% | *(Pending manual run)* | -- |
| **Abnormality Accuracy** | 11.27% | *(Pending manual run)* | -- |

## Comparison Protocol
Results from this diagnostic run will be directly compared against Split 1 of the official Image-Level reproduction (`run4/split_1`). Difference will be computed as:
$$\Delta = \text{QA Diagnostic Metric} - \text{Image Run 1 Metric}$$

## Interpretation Guidelines
* If Modality accuracy increases substantially (e.g., towards the paper's 76%), this supports the hypothesis that image familiarity across QA pairs facilitates modality recognition.
* If performance across categories remains largely unchanged, it indicates that split granularity is not the primary driver of the modality gap.
* Regardless of the outcome, we do **not** claim that the paper used QA-level splitting; the paper leaves the split unit unspecified.

## Conclusion
This experiment isolates the impact of split granularity under controlled experimental conditions. It provides empirical evidence regarding how evaluation protocol choices impact benchmark performance in medical VQA tasks.
