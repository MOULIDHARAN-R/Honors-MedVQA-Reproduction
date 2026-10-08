# Experiment: 5-Split Border Energy Shortcut Analysis

## Overview
This experiment investigates whether the convolutional feature representations (VGG-16 `features[28]`) focus on non-anatomical scan borders, orientation tags, or corner text as diagnostic classification shortcuts.

* **Metric**: Border Energy Ratio = proportion of Grad-CAM activation energy in the outer 10% perimeter frame vs. the central anatomical area.
* **Scope**: Evaluated across all 5 reproduction splits ($15,209$ total test instances).

---

## Key Results & Critical Statistical Interpretation

1. **Equal Border Energy in Correct vs. Incorrect Answers**:
   * Border energy is roughly equal between correct ($18.5\%$) and incorrect ($18.9\%$) predictions.
   * **Crucial Caveat**: Equal border energy does **not** disprove shortcut learning. In deep learning, shortcut features often produce the *correct* answer for the wrong reason (e.g. associating a specific scanner border with a specific imaging modality).
2. **Category Confounding**:
   * The global comparison is heavily confounded by question category. Abnormality has both the highest border energy and the highest error rate.
   * Within-category differences point in opposite directions:
     * Modality: $-1.95 \pm 0.68$ percentage points
     * Organ System: $-1.19$ percentage points
     * Abnormality: $-1.35$ percentage points
     * Plane: $+1.88$ percentage points
3. **Absence of Evidence $\ne$ Proof of Absence**:
   * A non-significant $p$-value indicates an absence of evidence for simple border shortcuts, not positive proof that shortcuts do not exist.

---

## How to Run

```powershell
python experiments/shortcut_analysis/run_shortcut_analysis.py --run run4
```
Results and plots are saved to `experiments/shortcut_analysis/results/` and `results/reproduction/run4/`.
