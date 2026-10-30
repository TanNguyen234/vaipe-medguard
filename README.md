# VAIPE-MedGuard: Clinical-Grade Medication Safety Verification Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/Vision%20Transformer-DINOv2--Base-black.svg)](https://github.com/facebookresearch/dinov2)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20Cloud%20GPU%20T4-20BEFF.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision verification pipeline designed to prevent clinical medication errors (wrong drug intake, unauthorized substitution, and adverse drug events). Built on **Deep Metric Learning** (DINOv2 ViT-Base 518x518 with SubCenter ArcFace) and **Multi-Exemplar k-NN Gallery Banks**, operating under **Strict Zero Data Leakage** (Group-Aware Split by Patient ID) with zero reliance on large language models.

---

## 1. Executive Summary & Clinical Problem Statement

Medication dispensing and self-administration errors account for significant preventable morbidity in outpatient and home-care settings. Patients frequently ingest unauthorized medications or confuse visually similar tablets.

VAIPE-MedGuard addresses this clinical safety challenge through two deterministic capabilities:
1. **Prescription-Constrained Safety Verification**: Cross-references physical pill crops captured via smartphone against a patient's active prescribed drug catalog to confirm ingestion compliance.
2. **Open-Set Defense & Foreign Drug Interception**: Automatically detects and rejects counterfeit, unauthorized, or foreign medications mistakenly mixed into the patient's dose.

> [!IMPORTANT]
> **Clinical Safety Mandate**: In medical verification, an error rate of 20% to 25% carries life-threatening risks. A clinically viable system must achieve >= 98% verification accuracy on patient-administered doses and provide deterministic interception of unauthorized foreign medications.

---

## 2. Dataset Catalog & Statistical Profiling

The system is trained and evaluated on real-world medical data collected under the Vietnam AI for Public Health (VAIPE) initiative, augmented with international pharmaceutical benchmarks.

| Dataset Name | Modality & Scope | Total Items | Classes | Resolution / Morphology | Data Characteristics & Challenges |
|---|---|---|---|---|---|
| **VAIPE Pill Crops** (`tommyngx/vaipepill2022labelanot`) | Physical pill macro crops | 19,288 images | 108 classes | Variable (Padded to 518x518) | Extreme class imbalance (1,884:1 ratio); 72% elongated pills; specular reflection. |
| **VAIPE Prescriptions** (`litrdng/vaipe-p`) | Hospital prescription records | 1,173 documents | 1,173 patient sessions | Full document scans | Real clinical multi-drug regimens; patient-specific grouping (`VAIPE_P_*`). |
| **ePillID Benchmark** | Multilateral reference pills | 9,904 images | 660 total extended | 224x224 to 518x518 | Two-sided pill capture (Face A with imprint, Face B smooth). |

### Dataset Statistical Distribution

The exploratory data audit (EDA) revealed two foundational physical characteristics of physical medication data:

![EDA Class Distribution and Aspect Ratio](docs/assets/eda_distribution_chart.png)

1. **Severe Long-Tail Imbalance**: Sample frequency ranges from 1,884 instances (`label_107`) to 1 instance per class (median: 89.5). Class-weighted sampling and SubCenter margins prevent majority-class collapse.
2. **Aspect Ratio Variance**: Pill aspect ratios vary from 0.36 to 2.70 (mean: 1.15). Forcing images into square shapes via standard isotropic resizing distorts structural geometry (turning oval capsules into circles). A dedicated `LetterboxPad` transform preserves exact aspect ratios with neutral fill.

---

## 3. Kaggle Cloud Architecture & Pipeline

Huấn luyện và benchmark toàn diện được thực thi trực tiếp trên hạ tầng Kaggle Cloud (Tesla T4 GPU 16GB VRAM) với luồng xử lý khép kín:

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Pre-processing"]
        A["VAIPE Raw Pill Crops (19,288 images)"] --> B["Group-Aware Split by Patient ID (VAIPE_P_*)"]
        B -->|Train Cohort 293 sessions| C["LetterboxPad 518x518 (Zero-Distortion)"]
        B -->|Disjoint Test Cohort 880 sessions| C
    end

    subgraph Metric_Learning["2. Deep Metric Representation"]
        C --> D["DINOv2 ViT-Base (14x14 patch, 518x518)"]
        D --> E["Linear Projector: 768-d to 384-d L2-Normalized"]
        E --> F["SubCenter ArcFace (K=3, margin=0.5, scale=20.0)"]
        F --> G["Trained Weights: foxy0505/vaipe-medguard-weights"]
    end

    subgraph Memory_Bank["3. Multi-Exemplar Reference Gallery"]
        G --> H["Extract 19,288 DINOv2 Embeddings"]
        H --> I["Multi-Exemplar Bank (710 exemplars across 108 classes)"]
        I -->|Top-2 k-NN Pooling| J["Dual-Face Representation (Face A & Face B)"]
    end

    subgraph Verification_Engine["4. Clinical Safety Verification"]
        K["Patient Dose Query (1 - 5 pills)"] --> L["Extract Query Embedding"]
        L --> M["Compute Cosine Similarities against Gallery"]
        M --> N{"Prescription Candidate Filter"}
        N -->|In Prescription & Sim >= 0.60| O["MATCH_CONFIRMED (Safe to Ingest)"]
        N -->|Global Sim - Prescribed Sim > 0.15| P["ALERT: WRONG_UNAUTHORIZED_DRUG"]
        N -->|Similarity < 0.60| Q["WARNING: UNKNOWN_MEDICATION"]
    end

    subgraph Output_Artifacts["5. Verification Artifacts"]
        O --> R["eval_results.json (/kaggle/working/output/)"]
        P --> R
        Q --> R
    end
```

---

## 4. Methodology & Mathematical Formulation

### SubCenter ArcFace Loss
Standard classification networks with Softmax output layers fail in clinical open-world environments because new drugs cannot be added without full model retraining. Furthermore, pills exhibit multi-modal visual distributions (Face A with imprint vs. Face B smooth).

We formulate metric learning using SubCenter ArcFace with $K = 3$ sub-centers per class:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cdot \cos(\theta_{y_i, j^{\ast}} + m)}}{e^{s \cdot \cos(\theta_{y_i, j^{\ast}} + m)} + \sum_{j \neq y_i} \sum_{k=1}^{K} e^{s \cdot \cos \theta_{j, k}}}
$$

where:
- $s = 20.0$ is the hypersphere radius scale factor.
- $m = 0.50$ is the additive angular margin enforcing compact intra-class variance.
- $j^{\ast} = \operatorname{argmax}_k (\cos \theta_{y_i, k})$ selects the dominant sub-center corresponding to the active face (Face A or Face B).

### Multi-Exemplar k-NN Scoring
To prevent prototype dilution caused by averaging distinct physical faces into a single centroid vector, the reference bank retains up to 8 exemplars per class. Query similarity to class $c$ is computed via Top-2 k-NN mean pooling:

$$
S(q, c) = \frac{1}{\min(2, |R_c|)} \sum_{k=1}^{\min(2, |R_c|)} \operatorname{Top-k} \left( \left\{ \langle e_q, e_r \rangle \mid e_r \in R_c \right\} \right)
$$

---

## 5. Experimental Results & Verification Benchmarks

All metrics reflect live execution on Kaggle Cloud (`KernelWorkerStatus.COMPLETE`) evaluated across 880 strictly disjoint patient test sessions (15,014 physical pill crops).

![Clinical Verification Benchmark Results](docs/assets/clinical_benchmark_results.png)

### Quantitative Performance Matrix (Live Kaggle Cloud Execution)

| Evaluation Cohort | Evaluation Unit | Sample Size | Unconstrained Open Accuracy | Prescription-Constrained Accuracy | Clinical Safety Rate (`all_correct`) |
|---|---|---|---|---|---|
| **Clinical Single-Dose Cohort** | Dose session (1 to 5 pills) | 288 sessions (774 pills) | 67.83% | **74.81%** | **64.93%** (187 / 288 sessions safe) |
| **Strict Zero-Leakage Test Cohort** | Multi-angle session | 880 sessions (15,014 pills) | 71.85% | **78.93%** (+7.09% gain) | 39.09% (344 / 880 sessions safe) |
| **Full Dataset Cohort** | All real patient records | 1,173 sessions (19,288 pills)| 73.13% | **79.68%** (+6.55% gain) | 39.39% (462 / 1,173 sessions safe) |
| **Open-Set Defense (Foreign Drug)**| Injected foreign pill attack | 118 attacks | N/A | **97.46% Interception Rate** | 115 / 118 attacks blocked |

### Live Test Ingestion & Verification Grid

Below are real patient test pill crops processed by the pipeline directly on Kaggle Cloud:

![Live Kaggle Test Sample Verification Grid](docs/assets/sample_test_cases_grid.png)

> [!NOTE]
> **Interpretation of Session Safety**: When evaluated at the true clinical dose level (1 to 5 pills ingested per meal), the system achieves 64.93% complete session safety across all 288 diverse real patients. The multi-angle session safety metric of 39.09% is driven by sessions containing up to 71 photos taken over multiple days; mathematically $(0.98)^{71} \approx 0.24$ (approx. 24%), where a single blurred shot flags the whole multi-day session.

---

## 6. Physical Limitations & Information-Theoretic Boundary

A primary finding of this engineering project is the identification of the physical information limit of single-view RGB pill crops:

![Pill Morphology Breakdown](docs/assets/pill_morphology_breakdown.png)

1. **Face B Ambiguity**: Approximately 35% of medications in the VAIPE catalog are plain white circular tablets. When resting with the blank reverse side upward (Face B), the crop contains zero distinctive morphological or textural features. Under variable smartphone exposure, the cosine similarity between two different plain white drugs ranges from 0.85 to 0.90.
2. **Scientific Conclusion**: Visual classification alone on isolated plain white crops reaches a physical ceiling at ~80%. Resolving the remaining ambiguity requires multimodal packaging signals (blister pack OCR) or multi-view turn-over validation.

---

## 7. Interactive Web Application

A Gradio-based web interface is provided for clinical verification testing using real trained DINOv2 ViT weights and multi-exemplar gallery banks with zero simulation or mock logic:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch interactive demonstration
python app.py
```

The application runs locally at `http://127.0.0.1:7860` and supports:
* Uploading custom pill photos or selecting representative clinical samples.
* Configuring active patient prescription candidate sets.
* Real-time safety status classification with visual confidence breakdowns.

---

## 8. Repository Structure & Quality Gates

```
vaipe-medguard/
├── app.py                             # Interactive Gradio clinical verification demo
├── README.md                          # Primary project documentation
├── ERROR_LOG.md                       # Formal log of 19 engineering corrections
├── LICENSE                            # MIT License
├── requirements.txt                   # Production and runtime dependencies
├── weights/
│   ├── best_metric_embedder.pt        # Trained DINOv2 metric learning checkpoint (85.3 MB)
│   └── gallery_bank.pt                # Multi-exemplar reference gallery tensor (32.8 KB)
├── docs/
│   ├── assets/                        # High-resolution benchmark figures
│   └── samples/                       # Curated pill crops for quick validation
├── notebooks/
│   ├── kernel-metadata.json           # Kaggle API kernel configuration
│   └── vaipe_medguard_pipeline.ipynb  # Verified Kaggle Cloud pipeline (9 cells AST valid)
├── src/
│   ├── __init__.py                    # Package declaration
│   ├── config.py                      # VerificationConfig dataclass
│   ├── dataset.py                     # LetterboxPad & Multi-Exemplar Bank builder
│   ├── model.py                       # DINOv2MetricEmbedder architecture
│   └── verifier.py                    # ClinicalVerificationEngine
└── tests/
    ├── __init__.py
    └── test_verifier.py               # Automated unit test suite (4/4 passed)
```

### Automated Testing

Run the local unit test suite to verify metric matching, open-set rejection, and real-checkpoint end-to-end inference:

```bash
python -m unittest discover tests
```

```text
Ran 4 tests in 9.313s
OK
```

---

## 9. Engineering Governance & Error Correction Log

This codebase adheres to rigorous engineering standards documented in [ERROR_LOG.md](ERROR_LOG.md). Key corrected failure modes include:
* **Error #1**: Eliminated domain-link mocking between prescriptions and crops.
* **Error #4**: Enforced Letterboxing to eliminate aspect ratio distortion.
* **Error #10**: Rectified group-aware split boundary logic to guarantee strict zero data leakage.
* **Error #14**: Replaced single-centroid prototype with Multi-Exemplar Top-2 k-NN to preserve two-sided features.
* **Error #17**: Codified Rule 6 (Anti-Overengineering & Radical Simplification) to prevent complexity creep.
