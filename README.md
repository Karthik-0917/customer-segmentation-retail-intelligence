# Customer Segmentation and Retail Intelligence using K-Means Clustering

An end-to-end, fully reproducible machine learning project that segments **4,338 retail customers** from the UCI Online Retail dataset into actionable behavioral groups using **RFM (Recency–Frequency–Monetary) analysis** and **K-Means clustering** — complete with a deterministic model-selection procedure, versioned artifacts, a strict inference contract, an interactive Streamlit dashboard, an automated test suite, and CI on GitHub Actions.

**Author:** Karthik Neduri · [karthikneduri17@gmail.com](mailto:karthikneduri17@gmail.com) · [GitHub: Karthik-0917](https://github.com/Karthik-0917)

**Repository:** https://github.com/Karthik-0917/customer-segmentation-retail-intelligence

---

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Solution Architecture](#solution-architecture)
5. [Data Cleaning](#data-cleaning)
6. [Feature Engineering](#feature-engineering)
7. [Model Selection — Deterministic K Procedure](#model-selection--deterministic-k-procedure)
8. [Results — Customer Segments](#results--customer-segments)
9. [Interactive Dashboard](#interactive-dashboard)
10. [Inference Contract](#inference-contract)
11. [Testing & Code Quality](#testing--code-quality)
12. [Reproducibility & Engineering Practices](#reproducibility--engineering-practices)
13. [Project Structure](#project-structure)
14. [Setup & Usage](#setup--usage)
15. [Key Artifacts](#key-artifacts)
16. [Limitations & Honest Caveats](#limitations--honest-caveats)
17. [Tech Stack](#tech-stack)
18. [Author](#author)

---

## Project Highlights

- **End-to-end ML pipeline**: raw Excel → cleaning → RFM feature engineering → model selection → trained artifacts → dashboard & CLI inference, all from a single command (`python -m src.pipeline`).
- **Deterministic, evidence-based K selection** — the number of clusters is *never* hard-coded. It is chosen by a documented procedure combining silhouette score, seed-stability (Adjusted Rand Index across a fixed seed set), and minimum cluster share, with explicit eligibility rules and tie-breaking.
- **Stability-tested clustering**: every candidate K (2–8) is re-fit under the fixed seed set `[42, 7, 21, 52, 101]`; mean ARI vs the seed-42 reference run is **0.9955** for the selected model — the segmentation is essentially seed-invariant.
- **Versioned model artifacts** (`model_version: rfm-kmeans-v1`, `feature_schema_version: 1.0.0`) with startup compatibility validation in the dashboard.
- **Strict inference contract**: schema validation, per-feature constraints (Recency ≥ 0, Frequency ≥ 1, Monetary > 0), NaN/inf rejection, and **out-of-distribution warnings** derived from persisted training ranges.
- **7-figure business context**: £8,887,208.89 in analyzed revenue across 18,532 orders, with per-segment revenue-share analysis.
- **Quality engineering**: comprehensive pytest suite on synthetic fixtures (no dataset needed to test), Ruff linting, Black formatting, and a GitHub Actions CI matrix on **Python 3.11 and 3.12**.

---

## Business Problem

Retail businesses cannot treat all customers identically — marketing budget, retention effort, and service levels should follow customer value and behavior. This project answers:

> *"Given one year of transaction history, which distinct behavioral groups exist in the customer base, how large and how valuable is each group, and how can a new customer's behavioral profile be assigned to a segment?"*

The output is a small set of interpretable segments with quantified size, spend, and activity profiles that can directly drive differentiated marketing and retention strategies.

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail (Dataset #352)](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
- **Content:** All transactions between **2010-12-01 and 2011-12-09** for a UK-based online retailer of unique all-occasion gifts.
- **Raw size:** 541,909 rows × 8 columns (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).
- The raw file (`data/Online_Retail.xlsx`, ~23 MB) is **deliberately not committed** to this repository. Download it from UCI and place it at `data/Online_Retail.xlsx` (see [Setup & Usage](#setup--usage)).

---

## Solution Architecture

```
Raw Excel (541,909 rows)
        │
        ▼
┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  Data Cleaning   │──▶│ RFM Feature Build │──▶│  Scaling (log1p +  │
│  (5-step audit)  │   │ (1 row/customer)  │   │  StandardScaler)   │
└─────────────────┘   └──────────────────┘   └────────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
┌──────────────────────────┐   ┌──────────────────────┐
│  K sweep (K = 2 … 8)      │──▶│ Deterministic K       │
│  5 fixed seeds per K      │   │ selection procedure   │
└──────────────────────────┘   └──────────────────────┘
                                          │
        ┌─────────────────────────────────┘
        ▼
┌──────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│  Final K-Means    │──▶│ Segment profiling  │──▶│ Versioned artifacts │
│  fit + naming     │   │ + business insights│   │ (models/, outputs/) │
└──────────────────┘   └───────────────────┘   └────────────────────┘
                                                        │
                              ┌─────────────────────────┴───────────┐
                              ▼                                     ▼
                   ┌────────────────────┐               ┌────────────────────┐
                   │ Streamlit dashboard │               │  CLI inference      │
                   │ (artifacts only —   │               │  (schema-validated, │
                   │  never retrains)    │               │  OOD-aware)         │
                   └────────────────────┘               └────────────────────┘
```

---

## Data Cleaning

A five-step auditable cleaning process, with every removal counted and persisted to `outputs/data_quality.json`:

| Step | Rule | Rows removed |
|---|---|---:|
| 1 | Exact duplicate rows | 5,268 |
| 2 | Missing `CustomerID` (cannot attribute to a customer) | 135,037 |
| 3 | Invalid invoice dates | 0 |
| 4 | Cancellations (`InvoiceNo` starting with "C") | 8,872 |
| 5 | Non-positive quantity or unit price | 40 |
| — | **Valid rows retained** | **392,692** |

**Resulting analysis base:** 4,338 customers · 18,532 valid orders · **£8,887,208.89** total revenue · average order value **£479.56** · reference date **2011-12-10** (max valid invoice date + 1 day — no information leakage past the observation window).

---

## Feature Engineering

One row per customer (CustomerID is an identifier only — **never** a clustering feature):

| Feature | Definition |
|---|---|
| **Recency** | Days from a customer's last purchase to the reference date |
| **Frequency** | Number of distinct valid orders (unique `InvoiceNo`) |
| **Monetary** | Total spend (Σ Quantity × UnitPrice) |

**Why log1p + StandardScaler?** Raw RFM features are heavily right-skewed, which distorts K-Means' Euclidean distances. The `log1p` transform reduces skewness dramatically before z-scaling:

| Feature | Skew (raw) | Skew (log1p) |
|---|---:|---:|
| Recency | 1.246 | −0.467 |
| Frequency | 12.067 | 1.209 |
| Monetary | 19.339 | 0.397 |

**Why only 3 features?** Candidate augmentations (AvgOrderValue, UniqueProducts, TotalQuantity, ActiveDays) were evaluated and rejected: they are strongly rank-correlated with the core RFM trio (Spearman: Monetary–TotalQuantity **0.928**, Frequency–ActiveDays **0.893**, Frequency–Monetary **0.807**) and added redundancy without improving cluster quality at any K. Country is used for **profiling only** — never one-hot encoded into the distance space.

---

## Model Selection — Deterministic K Procedure

The final K is **not hard-coded anywhere**. It is the output of a documented, reproducible procedure implemented in `src/clustering.py::select_k`:

1. **Sweep** K = 2 … 8. For each K, fit K-Means (`n_init=10`) under the fixed seed set **[42, 7, 21, 52, 101]**; seed 42 is the deterministic reference run (a reference — *not* ground truth).
2. **Evidence per K:** silhouette score, mean **ARI vs the seed-42 reference** across seeds, and smallest-cluster share.
3. **Evidence score:** equal-weight mean of the three metrics after min–max normalization across candidates.
4. **Eligibility:** at least 3 segments (a two-way split is too coarse for differentiated customer strategy) and smallest cluster ≥ 5% of customers.
5. **Tie-breaking (deterministic):** higher silhouette → higher stability → lower K.

**Actual results** (from `outputs/model_evaluation.csv`, produced by execution):

| K | Silhouette | Mean ARI vs seed 42 | Smallest cluster share | Evidence score | Status |
|---|---:|---:|---:|---:|---|
| 2 | 0.4340 | 0.9979 | 38.1% | 1.000 | Quantitative leader (below granularity minimum) |
| **3** | **0.3381** | **0.9955** | **17.2%** | **0.541** | ✅ **Selected** |
| 4 | 0.3368 | 0.9867 | 16.1% | 0.503 | Eligible |
| 5 | 0.3155 | 0.9408 | 6.7% | 0.229 | Eligible |
| 6 | 0.3143 | 0.9811 | 6.1% | 0.330 | Eligible |
| 7 | 0.3090 | 0.9932 | 5.0% | 0.339 | Eligible |
| 8 | 0.3021 | 0.8769 | 4.8% | 0.000 | Eligible |

**Transparent reasoning:** K=2 wins on raw quantitative evidence but reproduces only a coarse active-vs-inactive split, below the documented 3-segment business minimum. Among K ≥ 3 candidates, **K=3 beats K=4 on all three evidence components**. Nothing is hidden — the full table, including K=2, is preserved in the artifacts and displayed in the dashboard. The selected model's silhouette of 0.3381 indicates **moderate separation**, which is expected and acceptable for behavioral (non-spherical, continuous) customer data.

---

## Results — Customer Segments

**Selected model: K = 3** (silhouette 0.3381 · mean ARI vs seed 42: 0.9955):

| Segment | Customers | Share | Behavioral profile |
|---|---:|---:|---|
| 🟢 **Active High-Value Frequent Buyers** | 745 | 17.2% | Recent, frequent, high-spend customers — the revenue core |
| 🔵 **Steady Mid-Value Customers** | 1,677 | 38.7% | Moderately recent and regular; median ~31 days recency, ~£1,013 spend; 26.0% of revenue |
| 🟠 **Inactive Low-Spend Infrequent Customers** | 1,916 | 44.2% | Long time since last purchase, few orders, low spend |

Per-segment revenue vs customer-share comparison, standardized profile heatmaps (multipliers where **1.0× = population median**), and hypothesis-style business recommendations are generated into `outputs/business_insights.json` and `outputs/segment_profiles.csv`.

A PCA projection (2 components, **93.9%** variance explained: 75.2% + 18.7%) is used for **visualization only** — clustering always operates in the full scaled 3-D feature space.

---

## Interactive Dashboard

```bash
streamlit run app/predict.py
```

Built entirely from persisted artifacts — the dashboard **never retrains or recomputes** the model:

- **Sidebar model status:** model version, schema version, selected K, and artifact compatibility validation at startup.
- **Overview:** dataset KPIs, cleaning audit, monthly activity.
- **Segment explorer:** sizes, revenue share, standardized profile heatmap, deterministic per-segment colors used consistently on every chart.
- **Model analysis:** full K = 2–8 evidence table (K=2 tradeoff included), stability results, PCA cluster visualization with explicit "visualization only" caption.
- **Geography:** horizontal country chart with Customers / Revenue / AOV selector (profiling only).
- **Prediction:** interactive form with schema validation and out-of-distribution warnings; results are shown as an **Assigned Segment** (never as a confidence claim).
- **Methodology:** end-to-end pipeline diagram and honest documentation of every design decision.

---

## Inference Contract

Command-line inference against the trained artifacts:

```bash
python -m src.inference --recency 30 --frequency 5 --monetary 1500
```

Actual output:

```
Assigned segment : Cluster 2 — Steady Mid-Value Customers
Segment profile  : 1677 customers (38.7%), median recency 31 d, median 3 orders, median spend 1,013, revenue share 26.0%
Model            : rfm-kmeans-v1 (K=3, schema v1.0)
```

Guarantees enforced by `src/inference.py`:

- **No retraining** — the persisted scaler and K-Means model are loaded and applied as-is.
- **Schema validation** against `models/feature_schema.json` (feature names, order, count).
- **Hard constraints:** Recency ≥ 0, Frequency ≥ 1, Monetary > 0; NaN/inf rejected.
- **Out-of-distribution warnings** when an input falls outside the persisted training ranges — a transparency warning, not a rejection and not a fake confidence score.

---

## Testing & Code Quality

| Check | Command | Status |
|---|---|---|
| Unit tests | `python -m pytest -q` | ✅ All passing |
| Linting | `ruff check .` | ✅ All checks passed |
| Formatting | `black --check .` | ✅ 21 files, nothing to change |
| CI | GitHub Actions | ✅ Matrix: Python 3.11 & 3.12 |

The pytest suite covers data loading, cleaning rules, feature engineering, scaling, K selection determinism, segment profiling, artifact round-trips, and the full inference contract (validation, constraints, OOD warnings, corrupted-artifact rejection). **Tests run entirely on synthetic fixtures** — CI never needs the raw dataset.

---

## Reproducibility & Engineering Practices

- Single-command pipeline; all randomness pinned (`random_state=42`, explicit `n_init=10`, fixed stability seed set).
- Results verified **bit-identical across independent environments** (Windows / Python 3.14 local run vs Linux sandbox).
- Configuration externalized to `configs/config.yaml` — no magic numbers in code.
- No data leakage: the reference date is derived strictly from the observation window; features never look past it.
- No outlier deletion of high-value customers (they are legitimate business reality, handled via log-scaling instead).
- Raw dataset excluded from version control (`.gitignore`); artifacts under `models/` and `outputs/` are committed for exact result inspection.
- Honest metric language throughout: silhouette described as moderate separation; ARI measured against a named reference seed; no classification-accuracy claims for an unsupervised model.

---

## Project Structure

```
customer-segmentation/
├── .github/workflows/ci.yml     # CI: pytest + ruff + black on Python 3.11 & 3.12
├── app/
│   ├── predict.py               # Streamlit dashboard (entry point)
│   ├── charts.py                # Plotly chart builders (deterministic colors)
│   └── components.py            # Shared UI components, sidebar model status
├── configs/config.yaml          # All pipeline configuration (seeds, K range, paths)
├── data/                        # Place Online_Retail.xlsx here (gitignored)
├── models/                      # kmeans.joblib, scaler.joblib, feature_schema.json,
│                                # segment_mapping.json, model_metadata.json
├── outputs/                     # 12 generated artifacts: evaluation, stability,
│                                # profiles, insights, summary, parquet segments…
├── src/
│   ├── data_loader.py           # Excel/CSV loading with dtype discipline
│   ├── preprocessing.py         # 5-step audited cleaning
│   ├── features.py              # RFM build, scaling, schema, feature ranges
│   ├── clustering.py            # K sweep, stability, deterministic select_k
│   ├── profiling.py             # Segment naming, profiles, business insights
│   ├── pipeline.py              # End-to-end orchestration
│   ├── inference.py             # CLI inference with strict contract
│   └── config.py                # Typed access to configs/config.yaml
└── tests/                       # Pytest suite on synthetic fixtures
```

---

## Setup & Usage

```bash
# 1. Clone
git clone https://github.com/Karthik-0917/customer-segmentation-retail-intelligence.git
cd customer-segmentation-retail-intelligence

# 2. Virtual environment (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Dependencies
pip install -r requirements.txt

# 4. Dataset — download from UCI and place as:
#    data/Online_Retail.xlsx
#    https://archive.ics.uci.edu/dataset/352/online%2Bretail

# 5. Run the full pipeline (trains model, writes all artifacts)
python -m src.pipeline

# 6. Explore the dashboard
streamlit run app/predict.py

# 7. Assign a segment from the command line
python -m src.inference --recency 30 --frequency 5 --monetary 1500

# 8. Verify quality
python -m pytest -q
ruff check .
black --check .
```

> The dashboard and CLI run purely from committed artifacts, so steps 6–7 work even without the raw dataset.

---

## Key Artifacts

| File | Contents |
|---|---|
| `models/model_metadata.json` | Model version, selected K, seeds, training metadata |
| `models/feature_schema.json` | Feature names/order, constraints, training ranges (drives validation + OOD) |
| `outputs/model_evaluation.csv` | Full K = 2–8 evidence table with scores and statuses |
| `outputs/stability.json` | Per-K, per-seed ARI and silhouette results |
| `outputs/summary.json` | Selected K, selection reasoning, segment sizes, cleaning audit |
| `outputs/business_insights.json` | Per-segment revenue share and hypothesis-style recommendations |
| `outputs/customer_segments.parquet` | All 4,338 customers with features and assigned segments |

---

## Limitations & Honest Caveats

- **No churn claims.** The dataset contains no validated churn outcome, so no churn prediction or churn labeling is made anywhere — inactivity observations are exactly that: observations.
- Segment descriptions are **observational**; business recommendations are hypotheses to A/B test, not causal claims.
- Silhouette 0.3381 indicates **moderate** separation — typical for continuous behavioral data, and reported as such.
- Single retailer, single year, predominantly UK customers — segment structure may not transfer to other businesses.
- Inference assigns a profile from **behavioral history** (RFM); it cannot and does not derive behavior from demographics.

---

## Tech Stack

**Python 3.11+** · pandas · NumPy · scikit-learn · SciPy · Plotly · Matplotlib · Streamlit · PyArrow · openpyxl · PyYAML · joblib · pytest · Ruff · Black · GitHub Actions

---

## Author

**Karthik Neduri**

- 📧 [karthikneduri17@gmail.com](mailto:karthikneduri17@gmail.com)
- 💻 [github.com/Karthik-0917](https://github.com/Karthik-0917)

*Built as part of the InternsElite internship program.*

---

## License

See the [LICENSE](LICENSE) file for details.
