**CI:** ![CI](https://github.com/Karthik-0917/customer-segmentation-retail-intelligence/actions/workflows/ci.yml/badge.svg)
# Customer Segmentation and Retail Intelligence using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Project Overview

The **Customer Segmentation and Retail Intelligence** project is an end-to-end machine learning pipeline that segments retail customers into actionable behavioral groups using **RFM (Recency–Frequency–Monetary) analysis** and **K-Means clustering**.

The project processes one year of real transaction data from a UK-based online retailer, applies an auditable cleaning process, engineers per-customer behavioral features, selects the number of clusters through a deterministic evidence-based procedure, and presents the results through an interactive Streamlit dashboard and a schema-validated CLI inference tool.

**Objective:** Identify distinct behavioral customer groups from transaction history, quantify each group's size and revenue contribution, and provide a reliable mechanism to assign new customer profiles to segments.

---

## Intern Details

* **Name:** Karthik Neduri
* **Program:** InternsElite – AIML with Python Traineeship Program
* **Domain:** AI & Machine Learning / Unsupervised Learning
* **Project:** Customer Segmentation and Retail Intelligence using K-Means Clustering
* **Technology Focus:** Python, Machine Learning, Clustering, Data Analytics & Visualization
* **Date:** September 2026

---

## Dataset Information

* **Source:** [Online Retail Dataset – UCI Machine Learning Repository (Dataset #352)](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
* **Format:** Excel (.xlsx)
* **Size:** 541,909 rows × 8 columns
* **Time Period:** 01 December 2010 – 09 December 2011
* **Business:** UK-based online retailer of unique all-occasion gifts
* **Grain:** One row per invoice line item
* **Note:** The raw dataset (~23 MB) is not committed to this repository. Download it from UCI and place it at `data/Online_Retail.xlsx`.

### Dataset Columns

| Column | Description |
|---|---|
| InvoiceNo | Invoice number (prefix "C" indicates cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Quantity per transaction line |
| InvoiceDate | Date and time of the invoice |
| UnitPrice | Price per unit (GBP) |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Technology Stack

* **Programming Language:** Python 3.11+
* **Data Processing:** Pandas, NumPy, PyArrow, openpyxl
* **Machine Learning:** Scikit-learn (K-Means, StandardScaler, PCA, silhouette, ARI)
* **Statistical Analysis:** SciPy
* **Visualization:** Plotly, Matplotlib
* **Dashboard:** Streamlit
* **Model Persistence:** joblib
* **Configuration:** PyYAML
* **Testing:** Pytest
* **Code Quality:** Ruff, Black
* **Version Control:** Git, GitHub
* **CI/CD:** GitHub Actions (Python 3.11 & 3.12 matrix)

---

## Project Structure

```text
customer-segmentation-retail-intelligence/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── app/
│   ├── predict.py            # Streamlit dashboard (entry point)
│   ├── charts.py             # Plotly chart builders
│   └── components.py         # Shared UI components, sidebar model status
│
├── src/
│   ├── data_loader.py        # Excel/CSV loading with dtype discipline
│   ├── preprocessing.py      # 5-step audited cleaning
│   ├── features.py           # RFM features, scaling, schema, ranges
│   ├── clustering.py         # K sweep, stability, deterministic K selection
│   ├── profiling.py          # Segment naming, profiles, insights
│   ├── pipeline.py           # End-to-end orchestration
│   ├── inference.py          # CLI inference with strict contract
│   └── config.py             # Typed access to configuration
│
├── configs/
│   └── config.yaml           # Seeds, K range, paths — no magic numbers
│
├── models/                   # Versioned trained artifacts (committed)
├── outputs/                  # 12 generated analysis artifacts (committed)
├── tests/                    # Pytest suite on synthetic fixtures
├── data/
│   └── README.md             # Raw dataset placed here (gitignored)
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Installation & Usage

### Prerequisites

```bash
Python 3.11+
Git
```

### Clone the Repository

```bash
git clone https://github.com/Karthik-0917/customer-segmentation-retail-intelligence.git
cd customer-segmentation-retail-intelligence
```

### Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Place the Dataset

Download the Online Retail dataset from UCI and place it at:

```text
data/Online_Retail.xlsx
```

### Run the Training Pipeline

```bash
python -m src.pipeline
```

### Run the Dashboard

```bash
streamlit run app/predict.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

The repository includes the trained model artifacts (`models/`) and generated outputs (`outputs/`), so the dashboard and CLI inference can be used without re-running the pipeline.

### Run CLI Inference

```bash
python -m src.inference --recency 30 --frequency 5 --monetary 1500
```

### Run Quality Checks

```bash
python -m pytest -q
ruff check .
black --check .
```

---

## Key Analysis Areas

### Data Cleaning & Quality

* Five-step audited cleaning with every removal counted
* Duplicate removal (5,268 rows)
* Missing CustomerID handling (135,037 rows)
* Cancellation exclusion (8,872 rows)
* Non-positive quantity/price exclusion (40 rows)
* Persisted data-quality report (`outputs/data_quality.json`)

### Feature Engineering

* RFM features: one row per customer, CustomerID never used as a feature
* Skewness correction via log1p (Monetary skew reduced from 19.34 to 0.40)
* StandardScaler normalization for distance-based clustering
* Correlation-based rejection of redundant candidate features
* No data leakage past the reference date (max invoice date + 1 day)

### Model Selection & Stability

* K sweep from K=2 to K=8
* Fixed stability seed set [42, 7, 21, 52, 101] applied to every K
* Adjusted Rand Index measured against the seed-42 reference run
* Deterministic evidence-based K selection (no hard-coded K)
* Eligibility rules: minimum 3 segments, smallest cluster ≥ 5% of customers
* Deterministic tie-breaking: silhouette → stability → lower K

### Segment Profiling & Business Intelligence

* Automatic behavioral segment naming
* Per-segment revenue and revenue-share analysis
* Customer share vs revenue share comparison
* Standardized profile heatmap (1.0× = population median)
* Country-level profiling (profiling only — never a clustering feature)
* PCA 2-D cluster visualization (93.9% variance explained; visualization only)

### Inference Contract

* Strict schema validation against persisted feature schema
* Hard constraints: Recency ≥ 0, Frequency ≥ 1, Monetary > 0
* NaN/inf rejection
* Out-of-distribution warnings from persisted training ranges
* No retraining at inference time

---

## Key Results

### Data Processing Summary

| Metric | Value |
|---|---:|
| Raw Records | 541,909 |
| Valid Records After Cleaning | 392,692 |
| Unique Customers | 4,338 |
| Valid Orders | 18,532 |
| Total Revenue Analyzed | £8,887,208.89 |
| Average Order Value | £479.56 |
| Observation Window | 2010-12-01 → 2011-12-09 |

### Model Selection Evidence (K = 2–8)

| K | Silhouette | Mean ARI vs seed 42 | Smallest Cluster Share | Status |
|---|---:|---:|---:|---|
| 2 | 0.4340 | 0.9979 | 38.1% | Quantitative leader (below granularity minimum) |
| **3** | **0.3381** | **0.9955** | **17.2%** | ✅ **Selected** |
| 4 | 0.3368 | 0.9867 | 16.1% | Eligible |
| 5 | 0.3155 | 0.9408 | 6.7% | Eligible |
| 6 | 0.3143 | 0.9811 | 6.1% | Eligible |
| 7 | 0.3090 | 0.9932 | 5.0% | Eligible |
| 8 | 0.3021 | 0.8769 | 4.8% | Eligible |

K=2 leads quantitatively but reproduces only a coarse active-vs-inactive split, below the documented 3-segment business minimum. Among eligible candidates, K=3 wins on all three evidence components. The full table, including K=2, is preserved in the artifacts — nothing is hidden.

### Final Customer Segments (K = 3)

| Segment | Customers | Share | Profile |
|---|---:|---:|---|
| Active High-Value Frequent Buyers | 745 | 17.2% | Recent, frequent, high-spend — the revenue core |
| Steady Mid-Value Customers | 1,677 | 38.7% | Regular activity, ~£1,013 median spend, 26.0% of revenue |
| Inactive Low-Spend Infrequent Customers | 1,916 | 44.2% | Long inactivity, few orders, low spend |

Silhouette 0.3381 indicates moderate separation — expected and reported honestly for continuous behavioral data. Mean ARI of 0.9955 across the fixed seed set shows the segmentation is essentially seed-invariant.

---

## Statistical & Methodological Rigor

* **Silhouette score** – cluster separation quality per candidate K
* **Adjusted Rand Index (ARI)** – seed-stability measured against a named reference seed (a reference, not ground truth)
* **Min–max normalized evidence score** – equal-weight combination of silhouette, stability, and smallest-cluster share
* **Spearman rank correlation** – redundant feature elimination (Monetary–TotalQuantity 0.928, Frequency–ActiveDays 0.893)
* **Skewness analysis** – transformation validation before distance-based clustering

Methodological safeguards:

* No classification-accuracy language for an unsupervised model
* No outlier deletion of high-value customers (log-scaling used instead)
* Median-based profile descriptions (never calling a median an "average")
* Results verified bit-identical across two independent environments

---

## Dashboard Pages

The Streamlit dashboard runs entirely from persisted artifacts — it never retrains:

1. **Overview** – Dataset KPIs, cleaning audit, monthly activity
2. **Segments** – Sizes, revenue share, standardized profile heatmap
3. **Model Analysis** – Full K=2–8 evidence table, stability, PCA visualization
4. **Geography** – Country analysis with Customers / Revenue / AOV selector
5. **Prediction** – Interactive segment assignment with OOD warnings
6. **Methodology** – Pipeline diagram and documented design decisions

Sidebar shows global model status: model version (`rfm-kmeans-v1`), schema version, selected K, and artifact compatibility validation at startup.

---

## Business Insights

The pipeline converts segmentation results into practical recommendations:

* Prioritize retention investment in the 17.2% high-value segment that drives a disproportionate revenue share
* Design upgrade journeys for steady mid-value customers (largest revenue-growth headroom)
* Run low-cost reactivation experiments on the inactive segment before writing it off
* Compare customer share against revenue share to align budget with value, not headcount

> Findings are observational and associative. Recommendations are hypotheses to A/B test — not causal claims or guaranteed outcomes.

---

## Project Achievements

✅ Built an end-to-end customer segmentation pipeline on 541K+ real retail transactions

✅ Implemented a five-step audited data-cleaning process with persisted quality reports

✅ Engineered leakage-free RFM features with skewness-corrected scaling

✅ Designed a deterministic, evidence-based K-selection procedure (no hard-coded K)

✅ Validated stability with a fixed 5-seed protocol (mean ARI 0.9955)

✅ Shipped versioned model artifacts with schema validation and OOD-aware inference

✅ Developed an interactive multi-page Streamlit dashboard running purely from artifacts

✅ Implemented a comprehensive Pytest suite on synthetic fixtures (no dataset needed in CI)

✅ Enforced code quality with Ruff and Black

✅ Configured GitHub Actions CI on a Python 3.11 & 3.12 matrix

---

## Limitations & Honest Disclosure

* The dataset contains no validated churn outcome — no churn prediction or churn labeling is made anywhere; inactivity observations are exactly that.
* Silhouette 0.3381 indicates moderate (not strong) separation, typical for continuous behavioral data.
* Single retailer, single year, predominantly UK customers — segment structure may not transfer to other businesses.
* Segment descriptions are observational; business recommendations are hypotheses, not causal claims.
* Inference assigns a segment from behavioral history (RFM); it cannot derive behavior from demographics.

---

## Reports

* **Project Report:** Submitted separately as part of the InternsElite project submission
* **Technical Implementation:** Available in the source code
* **Dashboard:** Available through the Streamlit application
* **Source Code:** Available in this GitHub repository

---

## References

* [Online Retail Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
* [Python Documentation](https://docs.python.org/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
* [Plotly Documentation](https://plotly.com/python/)
* [Streamlit Documentation](https://docs.streamlit.io/)

---

## Author

**Karthik Neduri**

B.Tech Computer Science & Engineering
GITAM University — 2027

Email: [karthikneduri17@gmail.com](mailto:karthikneduri17@gmail.com)

GitHub: [https://github.com/Karthik-0917](https://github.com/Karthik-0917)

Project Repository:
[https://github.com/Karthik-0917/customer-segmentation-retail-intelligence](https://github.com/Karthik-0917/customer-segmentation-retail-intelligence)

---

⭐ If you found this project useful, consider giving the repository a star.

---

**InternsElite – AIML with Python Traineeship Program**
