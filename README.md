# Customer Segmentation and Retail Intelligence using K-Means Clustering

An end-to-end unsupervised machine learning system that transforms
transaction-level retail data into customer-level behavioral features,
discovers interpretable customer segments using K-Means clustering,
validates the segmentation using internal clustering diagnostics, and
provides interactive customer analysis and business decision support
through Streamlit.

---

## 1. Project Overview

Retailers accumulate large transaction logs but rarely have predefined
customer segments. This project takes the raw UCI Online Retail transaction
feed (~542k rows), applies a documented validation and cleaning policy,
aggregates it into one behavioral row per customer (RFM plus additional
behavioral features), selects a defensible feature set through
correlation/skewness diagnostics, clusters customers with K-Means using a
deterministic multi-criteria K-selection procedure, validates stability
across a fixed seed set, profiles and names the segments from their actual
statistics, and exposes the full analysis through a 7-page Streamlit
application and a strict, schema-checked inference CLI.

## 2. Business Problem

A retail company holds transaction-level purchase data but no customer
segments. Which customers are highly valuable? Who purchases frequently?
Whose purchasing activity has slowed? Which groups may warrant retention or
loyalty experiments? Clustering the customer base on observed purchasing
behavior yields descriptive groupings that support these questions. All
resulting recommendations are observational hypotheses — clustering does not
establish that any strategy causes revenue or retention changes.

## 3. Objectives

1. Convert raw transactions into a clean, reproducible customer-level dataset.
2. Engineer RFM and behavioral features with documented definitions.
3. Select features with evidence (redundancy, skewness, silhouette sweeps).
4. Choose K through a documented, deterministic multi-criteria procedure.
5. Validate stability across a fixed seed set applied to every candidate K.
6. Profile, name, and interpret segments from their actual statistics.
7. Persist a versioned, schema-consistent model for retraining-free inference.
8. Serve the analysis through a professional Streamlit application.

## 4. Dataset

- **Name:** Online Retail — UCI Machine Learning Repository (id 352)
- **URL:** <https://archive.ics.uci.edu/dataset/352/online+retail>
- **Structure:** transaction-level rows: `InvoiceNo`, `StockCode`,
  `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`,
  `Country` — a UK-based online gift retailer, December 2010 to December 2011.
- **Local path:** `data/Online_Retail.xlsx` (manual download; no account or
  API key required — see `data/README.md`).

## 5. Dataset License

Creative Commons Attribution 4.0 International (CC BY 4.0).
Attribution: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine
Learning Repository. <https://doi.org/10.24432/C5BW33>

The raw workbook is **gitignored**; only `data/README.md` is committed. All
derived artifacts are reproducible from the raw file via the training
pipeline.

## 6. Solution Architecture
UCI Online Retail (data/Online_Retail.xlsx)
→ Schema Validation (src/data_loader.py)
→ Transaction Cleaning + Cancellation Policy (src/preprocessing.py)
→ Valid Orders (unique InvoiceNo)
→ Customer Aggregation → RFM + Behavioral Features (src/features.py)
→ Skewness + Spearman Redundancy Diagnostics
→ Feature Selection → log1p → StandardScaler
→ K=2..8 Evaluation (elbow + silhouette + fixed-seed stability)
→ Deterministic Multi-Criteria K Selection (src/clustering.py)
→ Final K-Means (random_state=42, n_init=10)
→ Validation → Profiling → Data-Driven Segment Naming (src/profiling.py)
→ Revenue / Customer-Share Analysis
→ models/ artifacts + outputs/ analytical artifacts (src/pipeline.py)
→ Streamlit application (app/) + CLI inference (src/inference.py)

text


## 7. Data Cleaning

Counts produced by the actual pipeline run on the official UCI file:

| Step | Rows |
| --- | --- |
| Rows loaded | 541,909 |
| Exact duplicate rows removed | 5,268 |
| Missing `CustomerID` removed | 135,037 |
| Invalid dates removed | 0 |
| Cancellation invoices removed (`InvoiceNo` starting with "C") | 8,872 |
| Non-positive quantity/price removed (incl. bad-debt adjustments) | 40 |
| **Valid rows retained** | **392,692** |

Result: 18,532 valid orders across 4,338 customers; total valid revenue
£8,887,208.89. Every removal is counted dynamically and persisted to
`outputs/data_quality.json` — the dashboard displays generated values, never
hard-coded ones.

## 8. Transaction Definition

One **order/transaction** = one unique valid `InvoiceNo` after cleaning.
Multiple product rows on a single invoice count as **one** order. This
single definition is used consistently for Frequency, Average Order Value,
order counts, dashboard KPIs, and customer profiling. Revenue is
`Quantity × UnitPrice` on valid rows only.

## 9. Reference Date

`reference_date = max(valid InvoiceDate) + 1 day` — computed from the
cleaned data (2011-12-10 for this dataset). Recency therefore never depends
on the current system date, is identical on every rerun, and no information
after the reference date enters any feature (no future-information leakage).
Observation period: 2010-12-01 → 2011-12-09.

## 10. RFM

Computed per customer from valid purchases only:

- **Recency** — days between the customer's latest valid purchase and the
  reference date. *How recently the customer purchased.*
- **Frequency** — number of unique valid purchase invoices (orders).
  *How often the customer purchased.*
- **Monetary** — total valid purchase revenue for the customer.
  *How much valid revenue the customer generated.*

Together they provide three complementary dimensions of purchasing behavior.

## 11. Behavioral Features

Additional per-customer descriptors: `AvgOrderValue` (Monetary/Frequency),
`UniqueProducts` (distinct StockCodes), `TotalQuantity` (total units),
`ActiveDays` (days between first and last valid purchase), and `Country`
(modal country, descriptive only). These support profiling and the dashboard
but are **not** model inputs. `CustomerID` is retained solely as an
identifier and is never a clustering feature.

## 12. Feature Selection

| Feature | Definition | Candidate? | Used for clustering? | Transformation | Reason |
| --- | --- | --- | --- | --- | --- |
| Recency | Days since last valid purchase | Yes | **Yes** | log1p | Core engagement-recency axis |
| Frequency | Unique valid orders | Yes | **Yes** | log1p | Core purchase-rate axis |
| Monetary | Total valid revenue | Yes | **Yes** | log1p | Core value axis |
| AvgOrderValue | Monetary / Frequency | Yes | No | — | Ratio of two model features; lowered silhouette at every K in sweeps |
| UniqueProducts | Distinct products bought | Yes | No | — | Redundant with purchasing intensity; lowered silhouette at every K |
| TotalQuantity | Total units purchased | Yes | No | — | Near-duplicate of Monetary (Spearman ≈ 0.93) |
| ActiveDays | First-to-last purchase span | Yes | No | — | Near-duplicate of Frequency (Spearman ≈ 0.89) |
| Country | Modal customer country | No | No | — | Descriptive/profiling only; one-hot dummies would dominate the space |
| CustomerID | Identifier | No | Never | — | Not a behavioral attribute |

## 13. Redundancy Analysis

Spearman correlations (robust to skew) are computed on the actual dataset
and persisted to `outputs/correlation_matrix.csv`. Key measured values:
Monetary vs TotalQuantity ≈ 0.93, Frequency vs ActiveDays ≈ 0.89,
Frequency vs Monetary ≈ 0.81. Feature-set sweeps under identical
preprocessing showed every RFM-augmented candidate set producing lower
silhouette at every K, so the redundant candidates were excluded rather than
double-counting purchasing intensity.

## 14. Transformation

Measured raw skewness: Monetary 19.34, Frequency 12.07, AvgOrderValue 41.69
(see `outputs/feature_summary.csv`). `log1p` is applied to the three model
features, reducing skew to −0.47…1.21, so centroids are not dictated by a
handful of extreme customers. **Outliers are retained** — exceptionally
valuable customers are business signal, not noise; no records are deleted
and no capping is applied.

## 15. Standardization

`sklearn.preprocessing.StandardScaler` is fitted once on the log1p-transformed
customer feature matrix and persisted to `models/scaler.joblib`. Inference
loads and applies this exact scaler; no scaler is ever refitted outside the
training pipeline, and feature order is fixed by the persisted schema.

## 16. K-Means

`sklearn.cluster.KMeans` with explicit `random_state=42` and `n_init=10`
(no reliance on library defaults). K-Means is the final segmentation
algorithm; no alternative algorithm replaces it.

## 17. K Selection

Every K from 2 through 8 is evaluated. Reference-run (seed 42) metrics from
actual execution:

| K | Inertia | Silhouette | Smallest / largest cluster |
| - | ------- | ---------- | -------------------------- |
| 2 | 6488.0 | 0.4340 | 1,652 / 2,686 |
| 3 | 4879.1 | 0.3381 | 745 / 1,916 |
| 4 | 3962.7 | 0.3368 | 699 / 1,635 |
| 5 | 3331.9 | 0.3155 | 292 / 1,224 |
| 6 | 2875.4 | 0.3143 | 264 / 1,005 |
| 7 | 2573.2 | 0.3090 | 217 / 896 |
| 8 | 2359.6 | 0.3021 | 210 / 867 |

**Deterministic selection procedure** (implemented in
`src/clustering.py::select_k`, configured in `configs/config.yaml`):

1. **Eligibility filters:** a candidate K must produce at least 3 segments
   (a two-way active/inactive split is too coarse for differentiated
   customer strategy — this documented business requirement is why K=2,
   despite holding the highest raw silhouette, is excluded; the tradeoff is
   shown, not hidden) and every segment must hold at least 5% of customers
   (segments must be operationally addressable).
2. **Evidence score:** equal-weight mean of min-max-normalized silhouette,
   stability (mean ARI versus the seed-42 reference), and minimum cluster
   share, computed identically for every eligible K.
3. **Deterministic tie-break:** higher silhouette → higher stability →
   lower K.

The selected K, its full evidence table, and the generated "Why this K?"
explanation are produced by running the pipeline and persisted to
`models/model_metadata.json`, `outputs/model_evaluation.csv`, and
`outputs/summary.json`. The final model's inertia, silhouette, cluster
sizes, and segment names are recorded in those same generated artifacts —
they are outputs of the documented procedure applied to the actual data,
not hard-coded values. K=2..8 all remain visible in the Model Analysis page
and evaluation table.

## 18. Cluster Validation

- **Internal diagnostics:** inertia, silhouette score, cluster sizes and
  balance, centroid separation.
- **Fixed-seed stability:** the seed set `[42, 7, 21, 52, 101]` is applied
  identically to every candidate K with the same preprocessing, feature
  matrix, and `n_init`. Per K, the pipeline records mean/std of silhouette
  across seeds and Adjusted Rand Index of each seed's clustering **versus
  the seed-42 reference run** (a deterministic reference, not ground truth).
  Results are persisted to `outputs/stability.json`.
- No classification metrics (accuracy/precision/recall/F1) are used:
  clustering has no ground-truth labels, and no "accuracy" is claimed.

## 19. Cluster Profiles

For every cluster the pipeline computes customer count, customer share,
mean and median Recency/Frequency/Monetary, median AvgOrderValue, product
diversity, active duration, total revenue, and revenue share — persisted to
`outputs/segment_profiles.csv`. Medians are emphasized for the heavily
skewed monetary metrics, with means shown alongside. Segment medians are
compared against the population median (labelled explicitly as
"1.0× = population median" in the dashboard).

## 20. Business Insights

Per-segment insight blocks (generated into
`outputs/business_insights.json`) contain the observed profile, revenue
contribution, potential opportunity, potential risk, a strategy that could
be tested, and an explicit caution. All language is observational and
hypothesis-oriented ("this segment may be a candidate for…", "a controlled
experiment could test…"). No causal claims are made, and "churn" is not
used — the dataset contains no validated churn outcome; reduced recent
purchasing activity is described as exactly that.

## 21. Streamlit Application

Seven pages, all reading generated artifacts (never retraining, raw dataset
not required at runtime), with consistent segment names and deterministic
segment colors throughout:

1. **Executive Overview** — KPI cards (customers, valid orders, revenue,
   AOV, selected K, silhouette, observation period), segment distribution,
   revenue contribution by segment, customer share vs revenue share,
   monthly order/revenue trends, improved horizontal country chart with a
   Customers/Revenue/AOV metric selector.
2. **Customer Segments** — segment selector; profile cards (count, share,
   revenue, revenue share, median RFM/AOV/diversity/duration), population
   comparison heatmap, country table within the segment.
3. **Model Analysis** — elbow curve, silhouette by K (selected K
   highlighted, K=2 visible), candidate-K decision table, cluster-size
   chart with percentages, fixed-seed stability table, generated "Why this
   K?" explanation, PCA 2-D cluster visualization (visualization only —
   K-Means is trained on the full modeling feature space), standardized
   cluster profile visualization, and full model configuration from
   metadata.
4. **Customer Behavior** — segment and country filters, behavioral feature
   selector, distribution/boxplot/scatter charts with axis controls and
   mathematically safe log scaling.
5. **Segment Predictor** — assigns an **already-computed** customer
   behavioral profile (Recency/Frequency/Monetary) to the trained
   segmentation; validates inputs against the persisted schema; warns when
   inputs fall outside the observed training range (out-of-distribution
   warning, not a confidence claim); never retrains.
6. **Business Insights** — per-segment observational blocks with customer
   share and revenue share, framed as testable hypotheses.
7. **Data & Methodology** — dataset, license, cleaning table, transaction
   definition, reference date, RFM explanation, feature selection,
   modeling, K-selection procedure, limitations, and a methodology pipeline
   diagram.

A sidebar model-status panel shows the loaded model's K, features, version,
and training timestamp from metadata. Missing, corrupted, or
version-incompatible artifacts produce clear guidance messages instead of
tracebacks.

## 22. Inference

`src/inference.py` implements a strict training/inference contract:

- validates feature names, completeness (missing features rejected, never
  imputed), extras (rejected, never dropped), numeric types, finiteness,
  and per-feature constraints (Recency ≥ 0, Frequency ≥ 1, Monetary > 0)
  against `models/feature_schema.json`;
- applies the schema-defined log1p transform and the persisted training
  scaler in the persisted feature order;
- predicts with the persisted K-Means model and maps the cluster to its
  segment name and profile;
- emits an out-of-distribution warning when an input lies outside the
  observed training range recorded in model metadata;
- rejects artifact/schema/version incompatibilities with clear errors.

**The model is never retrained at inference.** RFM requires transaction
history: the tool assigns an existing/external customer's already-computed
behavioral profile — it does not derive RFM from demographics.

## 23. Model Artifacts

Generated by the training pipeline into `models/` (see `models/README.md`):

| Artifact | Contents |
| --- | --- |
| `scaler.joblib` | StandardScaler fitted on the log1p-transformed RFM matrix |
| `kmeans.joblib` | Final trained K-Means model |
| `segment_mapping.json` | cluster_id → segment name, naming rationales, profiles |
| `feature_schema.json` | Feature names, order, transformations, dtypes, constraints, schema version |
| `model_metadata.json` | Model version (`rfm-kmeans-v1`), schema version, dataset, observation period, reference date, features, selected K + selection evidence, seeds, validation metrics, training timestamp, library versions, observed feature ranges |

Derived analytical artifacts for the dashboard are written to `outputs/`
(summary, customer segments with PCA coordinates, segment profiles, model
evaluation, stability, correlation matrix, feature summary, population
comparison, business insights, monthly activity, country summary, data
quality). They are small, contain no raw transaction rows, and are
committed so the deployed dashboard runs without the raw dataset.

## 24. Reproducibility

Fixed `random_state=42` and `n_init=10`; fixed stability seed set
`[42, 7, 21, 52, 101]`; deterministic reference date and cleaning; a single
feature source of truth (`src/features.py` → `models/feature_schema.json`);
identical transform and scaler at training and inference; the documented
deterministic K-selection procedure applied identically on every rerun; and
`models/model_metadata.json` recording the exact configuration behind the
committed artifacts. Re-running the pipeline on the same dataset and
configuration reproduces the same selection and artifacts (the training
timestamp field is intentionally volatile and documented as such).

## 25. Project Structure
customer-segmentation/
├── app/
│ ├── init.py
│ ├── predict.py # 7-page Streamlit application
│ ├── components.py # cached artifact loading + error handling
│ └── charts.py # Plotly chart builders (consistent colors)
├── src/
│ ├── init.py
│ ├── config.py # paths + configs/config.yaml loading
│ ├── data_loader.py # load + schema validation
│ ├── preprocessing.py # cleaning policy, reference date, scaling
│ ├── features.py # RFM/behavioral features, diagnostics, schema
│ ├── clustering.py # K evaluation, stability, deterministic K selection
│ ├── profiling.py # profiles, naming, revenue analysis, insights
│ ├── inference.py # strict-contract CLI inference + OOD warnings
│ └── pipeline.py # training orchestrator
├── configs/
│ └── config.yaml # seeds, K range, selection rule parameters
├── data/
│ └── README.md # dataset source, license, setup (raw file gitignored)
├── models/ # generated model artifacts + README
├── outputs/ # generated dashboard/analytical artifacts
├── tests/ # synthetic-fixture test suite (no raw data needed)
├── .github/workflows/ci.yml # Ruff + Black + pytest on Python 3.11 / 3.12
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml

text


## 26. Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
27. Dataset Setup
Download the dataset zip from
https://archive.ics.uci.edu/dataset/352/online+retail (no account or API
key required), extract it, and save the workbook as:

text

data/Online_Retail.xlsx
28. Training
Bash

python -m src.pipeline
Runs the complete workflow (cleaning → features → K evaluation →
fixed-seed stability → deterministic K selection → training → profiling →
persistence) and regenerates all models/ and outputs/ artifacts.

29. Testing
Bash

python -m pytest -q
Tests run on small synthetic fixtures matching the UCI schema — the raw
dataset is not required.

30. Code Quality
Bash

python -m ruff check .
python -m black --check .
Continuous integration (.github/workflows/ci.yml) runs Ruff, Black, and
pytest on Python 3.11 and 3.12.

31. Streamlit Run
Bash

streamlit run app/predict.py
32. Inference Command
Bash

python -m src.inference --recency 30 --frequency 5 --monetary 1500
Arguments correspond exactly to the persisted feature schema: --recency
(days since last valid purchase, ≥ 0), --frequency (unique valid orders,
≥ 1), --monetary (total valid revenue, > 0).

33. Limitations
Single retailer and a single year of data (December 2010 – December
2011); findings do not generalize to other businesses or periods without
revalidation.
Roughly a quarter of raw rows lack CustomerID and are necessarily
excluded, so anonymous/guest purchases are under-represented.
The dataset does not provide a validated future churn outcome; therefore
the segmentation is evaluated primarily using internal clustering
diagnostics and descriptive behavioral differences.
Silhouette values in the observed range indicate moderate, not sharp,
separation — customer behavior is continuous and segment boundaries are
soft.
K-Means assumes roughly spherical clusters in the scaled space and
produces hard assignments; borderline customers may sit near two
segments.
The analysis is descriptive: it supports hypothesis generation, not
causal conclusions; proposed strategies require controlled experiments.
Monetary is gross purchase value (cancellations/returns excluded), not
net-of-returns value.
External inference is only as reliable as the supplied behavioral
features, which require genuine transaction history.
34. Future Improvements
Net-of-returns monetary view alongside the gross view.
Time-windowed retraining (e.g., quarterly) with segment-migration
tracking.
Gaussian Mixture Models as a soft-assignment diagnostic comparison.
Basket-level product-affinity features if reliable product categories can
be derived.
35. License
MIT License (see LICENSE). Dataset licensed under CC BY 4.0 and
attributed in data/README.md.

36. Author
Karthik Neduri


