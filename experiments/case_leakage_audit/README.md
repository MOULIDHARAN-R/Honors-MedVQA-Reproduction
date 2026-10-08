# Experiment: Patient & Case-Level Identifier Audit

## Overview
This diagnostic investigation checked whether the raw VQA-Med 2019 dataset contains a patient ID, case ID, study ID, or reliable grouping metadata that would allow partitioning the data at the patient level rather than image level.

---

## Findings

1. **Absence of Patient / Case Metadata**:
   * The VQA-Med 2019 dataset distribution provides only:
     - `image_id` (e.g. `synpic50385`)
     - `question`
     - `answer`
     - `category`
   * There is **no explicit patient ID, case ID, accession number, or study ID** in the dataset metadata or annotation files.
2. **Consecutive Image IDs Do Not Guarantee Patient Identity**:
   * An inspection of adjacent IDs (e.g. `synpic50385` and `synpic50387`) reveals visual similarities in some instances, but adjacent numbers are an artifact of indexing and cannot be mathematically assumed to belong to the same patient without external hospital accession metadata.
3. **Conclusion for honors thesis**:
   * Image-level splitting (ensuring 0% image overlap) is the strictest deterministic boundary that can be constructed from the public VQA-Med 2019 data. True patient-level splitting cannot be performed without external clinical records.

---

## How to Run

```powershell
python experiments/case_leakage_audit/run_case_audit.py
```
