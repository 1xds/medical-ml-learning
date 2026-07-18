# 📊 Medical Machine Learning Learning Notes

> NJMU · Summer 2026
>
> Personal learning notes based on the [ml4health](https://github.com/IILab-Resource/ml4health) tutorial — covering EDA, statistical testing, feature engineering, model evaluation, SHAP interpretability, causal inference, and radiomics.

---

**Status** 🚧 Ongoing Learning

---

## Objectives

- Build a systematic understanding of medical ML workflows.
- Learn to apply EDA, statistical testing, and feature engineering to clinical data.
- Understand model evaluation metrics (calibration, DCA) beyond accuracy.
- Master SHAP-based model interpretability for clinical decision support.
- Explore causal inference (DML) and radiomics pipelines.
- Develop reproducible computational workflows for medical data analysis.

---

## Workflow

ml4health Tutorial
        │
        ▼
Module-by-Module Notes (01–22)
        │
        ▼
Hands-on Practice Tasks (Task 01–07)
        │
        ▼
Integrated Medical ML Pipeline

---

## Research Highlights

- Completed EDA and statistical testing modules with real clinical datasets.
- Built a heart attack risk analysis pipeline (Task 01) using China-specific data.
- Covered the full SHAP interpretability series (Overview, Interaction, Dependence, Clustering, Decision Path, Bootstrap).
- Integrated DML causal inference and radiomics feature extraction into the learning path.

---

## Learning Progress

| # | Module | Status |
|---|--------|--------|
| 01 | [EDA](notes/01_EDA.md) | ✅ |
| 02 | [Statistical Tests](notes/02_Statistical_Tests.md) | ✅ |
| 03 | [Preprocessing](notes/03_Preprocessing.md) | ✅ |
| 04 | [Feature Construction](notes/04_Feature_Construction.md) | ✅ |
| 05 | [Feature Selection](notes/05_Feature_Selection.md) | ✅ |
| 06 | [Dimensionality Reduction & Clustering](notes/06_Dimensionality_Reduction_Clustering.md) | ✅ |
| 07 | [Data Leakage Prevention](notes/07_Data_Leakage_Prevention.md) | ✅ |
| 08 | [Cross Validation](notes/08_Cross_Validation.md) | ⬜ |
| 09 | [Model Comparison](notes/09_Model_Comparison.md) | ⬜ |
| 10 | [Imbalanced Data](notes/10_Imbalanced_Data.md) | ⬜ |
| 11 | [Calibration & DCA](notes/11_Calibration_DCA.md) | ⬜ |
| 12 | [SHAP Overview](notes/12_SHAP_Overview.md) | ⬜ |
| 13 | [SHAP Interaction](notes/13_SHAP_Interaction.md) | ⬜ |
| 14 | [SHAP Dependence](notes/14_SHAP_Dependence.md) | ⬜ |
| 15 | [SHAP Clustering](notes/15_SHAP_Clustering.md) | ⬜ |
| 16 | [SHAP Decision Path](notes/16_SHAP_Decision_Path.md) | ⬜ |
| 17 | [SHAP Bootstrap](notes/17_SHAP_Bootstrap.md) | ⬜ |
| 18 | [DML Causal Inference](notes/18_DML_Causal_Inference.md) | ⬜ |
| 19 | [Radiomics Feature Extraction](notes/19_Radiomics_Feature_Extraction.md) | ⬜ |
| 20 | [Radiomics ML Pipeline](notes/20_Radiomics_ML_Pipeline.md) | ⬜ |
| 21 | [TCGA Multiclass](notes/21_TCGA_Multiclass.md) | ⬜ |
| 22 | [CGM Glucose Regression](notes/22_CGM_Glucose_Regression.md) | ⬜ |

---

## Practice Tasks

Hands-on tasks mapped to prerequisite modules:

| Task | Name | Prerequisites | Status |
|------|------|--------------|--------|
| 01 | [EDA + Statistical Report](Task/Task%2001/) | 01 EDA, 02 Statistical Tests | ✅ |
| 02 | Feature Engineering Pipeline | 03 Preprocessing, 04 Feature Construction, 05 Feature Selection | 🔄 |

---

## Repository Structure

```
medical-ml-learning/
├── README.md            # Project overview
├── resources.md         # Reference materials and links
├── notes/               # Learning notes (01–22)
│   ├── 01_EDA.md
│   ├── 02_Statistical_Tests.md
│   ├── ...
│   └── 22_CGM_Glucose_Regression.md
└── Task/                # Hands-on practice tasks
    ├── Task 01/
    │   ├── heart_attack_analysis.ipynb
    │   ├── heart_attack_china.csv
    │   └── chart_interpretation.md
    ├── Task 02/
    ├── ...
    └── Task 07/
```

---

## Tools

- **Python** — pandas, numpy, matplotlib, seaborn, scikit-learn
- **SHAP** — Model interpretability (modules 12–17)
- **imbalanced-learn** — Class imbalance handling (module 10)
- **EconML / DoubleML** — Causal inference (module 18)
- **PyRadiomics** — Radiomics feature extraction (modules 19–20)
- **Jupyter Notebook** — Interactive analysis and practice tasks

---

## References

Key references and learning resources are listed in [`resources.md`](resources.md).

- [ml4health Tutorial](https://github.com/IILab-Resource/ml4health) — Primary learning source
- Lundberg SM, Lee SI. (2017) *NeurIPS* — A Unified Approach to Interpreting Model Predictions (SHAP)
- Chernozhukov V et al. (2018) *Econometrics Journal* — Double/Debiased Machine Learning (DML)

---

## Author

**1xds** — [GitHub](https://github.com/1xds)

NJMU Bioinformatics Graduate Student

Interested in Computational Biology, Structural Bioinformatics, and Medical AI.

---

## Future Work

- Complete remaining module notes (03–22)
- Finish all 7 practice tasks
- Build an integrated clinical prediction pipeline
- Develop reproducible computational workflows for medical data analysis
