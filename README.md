# Customer Segmentation and Retail Intelligence using K-Means Clustering

**Live Demo:** https://customer-segmentation-retail-intelligence.streamlit.app/  
**CI:** ![CI](https://github.com/Karthik-0917/customer-segmentation-retail-intelligence/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-K--Means-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## Project Overview

The **Customer Segmentation and Retail Intelligence** project is an end-to-end machine learning and analytics solution that segments retail customers into behavioral groups using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means clustering**.

The project processes one year of real transaction data from a UK-based online retailer. The workflow includes data validation, audited transaction cleaning, customer-level aggregation, RFM and behavioral feature engineering, skewness handling, feature selection, scaling, K-Means model selection, multi-seed stability analysis, segment profiling, and business interpretation.

The resulting customer segments are presented through an interactive **Streamlit retail intelligence application**, while a schema-validated CLI inference tool provides reproducible assignment of already-computed customer behavioral profiles to the trained segments.

**Objective:** Identify distinct behavioral customer groups from transaction history, quantify each group's size and revenue contribution, and provide a reproducible mechanism for assigning already-computed customer behavioral profiles to trained segments.

---

## Intern Details

- **Name:** Karthik Neduri
- **Program:** InternsElite – AIML with Python Traineeship Program
- **Domain:** AI & Machine Learning / Unsupervised Learning
- **Project:** Customer Segmentation and Retail Intelligence using K-Means Clustering
- **Technology Focus:** Python, Machine Learning, Clustering, Data Analytics & Visualization
- **Date:** September 2026

---

## Problem Statement

Retail businesses often have customers with very different purchasing behaviors. Treating the entire customer base as a single group can hide meaningful differences in purchase frequency, spending level, and recent activity.

The objective of this project is to transform transaction-level retail data into customer-level behavioral profiles and use unsupervised machine learning to identify meaningful customer segments.

The project focuses on:

- Cleaning and validating raw retail transactions
- Excluding invalid, cancelled, or incomplete transactions
- Aggregating transaction history to the customer level
- Engineering RFM and supporting behavioral features
- Handling skewed monetary behavior
- Selecting a suitable feature space for clustering
- Determining the number of clusters using documented evidence
- Testing clustering stability across multiple fixed random seeds
- Profiling the resulting customer segments
- Comparing customer share with revenue contribution
- Providing an interactive business intelligence dashboard
- Supporting reproducible segment inference from existing RFM profiles

---

## Dataset Information

### Online Retail Dataset

The project uses the **Online Retail Dataset** from the UCI Machine Learning Repository.

- **Source:** [UCI Machine Learning Repository – Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
- **Dataset ID:** 352
- **Format:** Excel (.xlsx)
- **Size:** 541,909 rows × 8 columns
- **Time Period:** 01 December 2010 – 09 December 2011
- **Business:** UK-based online retailer selling unique all-occasion gifts
- **Grain:** One row per invoice line item
- **Dataset License:** CC BY 4.0
- **Raw Dataset:** Not committed to the repository

The raw dataset should be downloaded from UCI and placed locally at:

```text
data/Online_Retail.xlsx
````

### Dataset Columns

| Column      | Description                                       |
| ----------- | ------------------------------------------------- |
| InvoiceNo   | Invoice number; prefix `C` indicates cancellation |
| StockCode   | Product code                                      |
| Description | Product name                                      |
| Quantity    | Quantity per invoice line                         |
| InvoiceDate | Invoice date and time                             |
| UnitPrice   | Price per unit in GBP                             |
| CustomerID  | Unique customer identifier                        |
| Country     | Customer country                                  |

---

## Technology Stack

| Category                 | Technology                       |
| ------------------------ | -------------------------------- |
| Programming Language     | Python 3.11+                     |
| Data Processing          | Pandas, NumPy, PyArrow, openpyxl |
| Machine Learning         | Scikit-learn                     |
| Clustering               | K-Means                          |
| Scaling                  | StandardScaler                   |
| Dimensionality Reduction | PCA                              |
| Statistical Analysis     | SciPy                            |
| Visualization            | Plotly, Matplotlib               |
| Dashboard                | Streamlit                        |
| Model Persistence        | joblib                           |
| Configuration            | PyYAML                           |
| Testing                  | Pytest                           |
| Code Quality             | Ruff, Black                      |
| Version Control          | Git, GitHub                      |
| Continuous Integration   | GitHub Actions                   |
| CI Matrix                | Python 3.11 and 3.12             |

---

## Project Structure

```text
customer-segmentation-retail-intelligence/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── app/
│   ├── predict.py            # Streamlit dashboard entry point
│   ├── charts.py             # Plotly chart builders
│   └── components.py         # Shared UI components
│
├── src/
│   ├── data_loader.py        # Dataset loading and validation
│   ├── preprocessing.py      # Audited transaction cleaning
│   ├── features.py           # RFM features, transformations and schemas
│   ├── clustering.py         # K sweep, stability and K selection
│   ├── profiling.py          # Segment profiles, names and insights
│   ├── pipeline.py           # End-to-end training orchestration
│   ├── inference.py          # CLI segment inference
│   └── config.py             # Typed configuration access
│
├── configs/
│   └── config.yaml           # Model, feature and path configuration
│
├── models/                   # Persisted trained model artifacts
│
├── outputs/                  # Generated analytical artifacts
│
├── tests/                    # Automated test suite
│
├── data/
│   └── README.md             # Dataset source and local placement instructions
│
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions CI workflow
```

The raw dataset file is intentionally excluded from version control.

---

## Installation & Usage

### Prerequisites

* Python 3.11+
* Git

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

Download the Online Retail dataset from the UCI Machine Learning Repository and place it at:

```text
data/Online_Retail.xlsx
```

### Run the Training Pipeline

The complete workflow is orchestrated through:

```bash
python -m src.pipeline
```

The pipeline performs:

```text
Raw Transactions
      ↓
Data Validation
      ↓
Transaction Cleaning
      ↓
Customer Aggregation
      ↓
RFM + Behavioral Features
      ↓
Feature Selection
      ↓
Log Transformation
      ↓
StandardScaler
      ↓
K-Means Candidate Evaluation
      ↓
Fixed-Seed Stability Analysis
      ↓
Deterministic K Selection
      ↓
Final K-Means Model
      ↓
Segment Profiling
      ↓
Business Insights
      ↓
Persisted Artifacts
```

### Run the Streamlit Dashboard

```bash
streamlit run app/predict.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

The dashboard reads persisted analytical and model artifacts. It does not retrain the clustering model during normal application use.

### Run CLI Inference

Example:

```bash
python -m src.inference --recency 30 --frequency 5 --monetary 1500
```

The inference tool applies the same persisted feature transformation, scaler, and K-Means model used during training.

RFM features must already be available from transaction history. The inference tool does not derive customer behavior from demographic information.

### Run Quality Checks

```bash
python -m pytest -q
python -m ruff check .
python -m black --check .
```

---

## Data Processing & Feature Engineering

### Data Cleaning

The raw transaction data is processed through an auditable cleaning workflow.

The current run records:

| Processing Step                          |    Rows |
| ---------------------------------------- | ------: |
| Raw records loaded                       | 541,909 |
| Exact duplicate rows removed             |   5,268 |
| Missing CustomerID rows removed          | 135,037 |
| Invalid date rows removed                |       0 |
| Cancellation invoices removed            |   8,872 |
| Non-positive quantity/price rows removed |      40 |
| Valid transaction records                | 392,692 |

Each cleaning stage is recorded so that the transformation from raw transactions to valid analytical records remains traceable.

### Transaction Definition

For customer-level behavioral analysis, a valid transaction is defined as a valid invoice-level purchase after cleaning.

Multiple invoice lines belonging to the same invoice are treated as one order for order-frequency calculations.

This definition is applied consistently across the training pipeline and analytical outputs.

---

## RFM Feature Engineering

RFM represents three complementary dimensions of customer purchasing behavior.

### Recency

Number of days since the customer's most recent valid purchase.

Lower values indicate more recent activity.

### Frequency

Number of unique valid purchase orders associated with the customer.

Higher values indicate more frequent purchasing.

### Monetary

Total valid revenue generated by the customer.

Higher values indicate greater historical spending.

The customer-level dataset therefore contains one row per customer rather than one row per transaction.

---

## Additional Behavioral Features

Additional behavioral measures are calculated for profiling and analytical exploration, including:

* Average Order Value
* Unique Products
* Total Quantity
* Active Days
* Country

These features support deeper segment interpretation and dashboard analysis.

The final clustering model uses the selected behavioral feature schema documented in the persisted model artifacts.

Country is used for profiling and geographic analysis but is not used as a clustering feature.

---

## Feature Selection & Transformation

CustomerID is never used as a model feature.

Candidate behavioral features are evaluated for redundancy before clustering.

The analysis identified strong relationships such as:

* Monetary vs TotalQuantity: Spearman correlation ≈ 0.928
* Frequency vs ActiveDays: Spearman correlation ≈ 0.893

Highly redundant variables are not unnecessarily included in the final clustering space.

Because monetary behavior is highly right-skewed, a `log1p` transformation is applied before scaling.

The recorded Monetary skewness was reduced from approximately:

```text
19.34 → 0.40
```

The transformed features are then standardized using `StandardScaler`.

This is important because K-Means is distance-based and differences in feature scale can disproportionately affect cluster assignment.

---

## Reference Date & Leakage Control

The reference date is derived deterministically from the transaction observation period.

The current project uses:

```text
Reference Date = Maximum valid InvoiceDate + 1 day
```

For the current dataset, the observation period ends on:

```text
2011-12-09
```

The resulting reference date is used consistently when calculating Recency.

The feature engineering process does not use information beyond the defined reference point.

---

## Model Selection

### K-Means Clustering

K-Means is used to partition customers into groups with similar behavioral profiles.

The candidate range is:

```text
K = 2 to 8
```

Each candidate is evaluated using:

* Inertia
* Silhouette score
* Minimum cluster size
* Cluster share
* Multi-seed stability
* Adjusted Rand Index

The model uses:

```text
random_state = 42
n_init = 10
```

---

## Deterministic K-Selection

The project does not hard-code the final number of clusters.

A deterministic multi-criteria procedure evaluates every candidate K.

The process considers:

1. Silhouette quality
2. Cluster-size sanity checks
3. Minimum segment granularity
4. Fixed-seed stability
5. Adjusted Rand Index
6. Smallest-cluster share
7. A documented evidence score
8. Deterministic tie-breaking

The fixed stability seed set is:

```text
[42, 7, 21, 52, 101]
```

The same seed set is applied to every candidate K.

K=2 produces the strongest raw silhouette score but results in only a coarse active-versus-inactive partition. The documented business granularity requirement therefore excludes K=2.

Among eligible candidates, K=3 is selected through the documented deterministic procedure.

No preferred K is hard-coded into the implementation.

---

## Key Results

### Data Processing Summary

| Metric                       |                   Value |
| ---------------------------- | ----------------------: |
| Raw Records                  |                 541,909 |
| Valid Records After Cleaning |                 392,692 |
| Unique Customers             |                   4,338 |
| Valid Orders                 |                  18,532 |
| Total Revenue Analyzed       |           £8,887,208.89 |
| Average Order Value          |                 £479.56 |
| Observation Window           | 2010-12-01 → 2011-12-09 |

---

## Model Selection Evidence

| K     | Silhouette | Mean ARI vs Seed 42 | Smallest Cluster Share | Status                                         |
| ----- | ---------: | ------------------: | ---------------------: | ---------------------------------------------- |
| 2     |     0.4340 |              0.9979 |                  38.1% | Quantitative leader, below granularity minimum |
| **3** | **0.3381** |          **0.9955** |              **17.2%** | **Selected**                                   |
| 4     |     0.3368 |              0.9867 |                  16.1% | Eligible                                       |
| 5     |     0.3155 |              0.9408 |                   6.7% | Eligible                                       |
| 6     |     0.3143 |              0.9811 |                   6.1% | Eligible                                       |
| 7     |     0.3090 |              0.9932 |                   5.0% | Eligible                                       |
| 8     |     0.3021 |              0.8769 |                   4.8% | Eligible                                       |

K=3 is the selected final model.

The full candidate table is preserved in the generated analytical artifacts.

---

## Final Customer Segments

The final K=3 solution produces three behavioral customer groups.

| Segment                                 | Customers | Customer Share | Profile                                                |
| --------------------------------------- | --------: | -------------: | ------------------------------------------------------ |
| Active High-Value Frequent Buyers       |       745 |          17.2% | Recent, frequent and high-spend customers              |
| Steady Mid-Value Customers              |     1,677 |          38.7% | Regular customers with moderate purchasing value       |
| Inactive Low-Spend Infrequent Customers |     1,916 |          44.2% | Long inactivity, few orders and lower historical spend |

### Revenue Contribution

| Segment                                 | Revenue Share |
| --------------------------------------- | ------------: |
| Active High-Value Frequent Buyers       |         66.3% |
| Steady Mid-Value Customers              |         26.0% |
| Inactive Low-Spend Infrequent Customers |          7.7% |

These figures describe observed historical behavior in the dataset and should not be interpreted as causal business effects.

---

## Model Stability

The final clustering solution was evaluated using a fixed five-seed protocol:

```text
42
7
21
52
101
```

For K=3:

```text
Mean Silhouette = 0.3381
Mean ARI vs Seed 42 = 0.9955
```

The high ARI indicates that the resulting partition is highly stable across the tested random seeds.

ARI is used as a stability measure against the deterministic seed-42 reference run. Seed 42 is a reference configuration, not ground truth.

---

## Statistical & Methodological Rigor

The project incorporates several safeguards to make the analysis reproducible and auditable.

### Evaluation Methods

* Silhouette score for cluster separation
* Inertia for within-cluster compactness
* Adjusted Rand Index for clustering stability
* Spearman rank correlation for redundancy analysis
* Skewness analysis for transformation validation
* PCA for two-dimensional visualization only

### Methodological Safeguards

* No classification accuracy is reported for the unsupervised model
* CustomerID is never used as a clustering feature
* Country is used for profiling, not clustering
* High-value customers are not removed simply because they are outliers
* Log transformation is used to reduce monetary skew
* Median values are reported as medians rather than being described as averages
* RFM features are calculated from transaction history
* Inference uses the persisted feature schema
* No retraining occurs during inference
* Out-of-distribution input warnings are supported
* No causal business claims are made from observational clustering results

---

## Visualization Portfolio

The project includes an interactive visualization layer covering both machine learning evaluation and retail business analysis.

### Executive Overview

* Customer base size
* Valid orders
* Total revenue
* Average order value
* Selected K
* Silhouette score
* Observation period
* Customers by segment
* Revenue contribution by segment
* Customer share vs revenue share
* Monthly order activity
* Monthly revenue
* Country-level customer distribution
* Country-level revenue distribution

### Customer Segments

* Segment selection
* Customer share
* Segment revenue
* Revenue share
* Median Recency
* Median Frequency
* Median Monetary
* Average Order Value
* Unique Products
* Active Days
* Relative-to-population segment profile heatmap
* Country distribution within segment

### Model Analysis

* Elbow curve
* Silhouette score by K
* Deterministic K-selection procedure
* Candidate K decision table
* Fixed-seed stability table
* Cluster-size comparison
* Segment profile heatmap
* PCA 2-D cluster visualization
* Model configuration and metadata

### Customer Behavior

Interactive exploration of:

* Recency
* Frequency
* Monetary
* Average Order Value
* Unique Products
* Total Quantity
* Active Days

The page supports:

* Segment filtering
* Country filtering
* Distribution analysis
* Box plots
* Scatter exploration
* Optional logarithmic axes where appropriate

### Business Insights

* Customer share vs revenue share
* Segment-level observations
* Potential strategic actions
* Risk/opportunity framing
* Hypothesis-oriented recommendations

---

## Streamlit Retail Intelligence Dashboard

The project provides a seven-page Streamlit application.

### 1. Executive Overview

Provides an executive-level summary of the customer base, revenue, orders, segment distribution, monthly trends, and geographic contribution.

### 2. Customer Segments

Provides detailed segment-level exploration, including behavioral KPIs, revenue contribution, relative profile comparisons, and country distributions.

### 3. Model Analysis

Provides complete model-selection evidence, including K=2–8 evaluation, stability analysis, ARI, cluster sizes, PCA visualization, and model configuration.

### 4. Customer Behavior

Provides interactive exploration of customer behavioral features across segments and countries.

### 5. Customer Segment Predictor

Assigns an already-computed RFM profile to a trained customer segment using the persisted transformation, scaler, and K-Means model.

RFM features require transaction history. The predictor does not derive customer behavior from demographic information.

### 6. Business Insights

Presents observed segment characteristics, customer/revenue concentration, and hypothesis-oriented business strategies.

### 7. Data & Methodology

Documents:

* Dataset source
* Data cleaning
* Transaction definition
* Reference date
* RFM methodology
* Feature engineering
* Feature selection
* K-selection procedure
* Stability methodology
* Limitations

The sidebar also displays model status information including:

* Model version
* Schema version
* Selected K
* Feature schema
* Training timestamp
* Artifact compatibility status

---

## Business Insights

The segmentation results provide an analytical basis for understanding customer concentration and behavioral differences.

### Active High-Value Frequent Buyers

This group represents approximately **17.2% of customers** while contributing approximately **66.3% of observed revenue**.

**Potential strategy to test:**

* Retention-focused programs
* Loyalty experiments
* Early-access or premium-service initiatives
* Personalized engagement

### Steady Mid-Value Customers

This is the largest revenue-producing group after the high-value segment, representing approximately **38.7% of customers** and **26.0% of revenue**.

**Potential strategy to test:**

* Cross-sell and upsell experiments
* Repeat-purchase incentives
* Product discovery campaigns
* Loyalty progression strategies

### Inactive Low-Spend Infrequent Customers

This group represents approximately **44.2% of customers** but contributes approximately **7.7% of revenue**.

**Potential strategy to test:**

* Low-cost reactivation campaigns
* Targeted offers
* Re-engagement messaging
* Controlled win-back experiments

> These findings are observational and associative. The recommended strategies are hypotheses for controlled testing and do not establish causal business impact.

---

## Processing & Reproducibility

The project is designed around a reproducible training workflow.

The primary training command is:

```bash
python -m src.pipeline
```

The pipeline is responsible for:

1. Loading the UCI Online Retail dataset
2. Validating the input structure
3. Cleaning transaction records
4. Defining valid transactions
5. Aggregating transactions to customer level
6. Computing RFM and behavioral features
7. Evaluating feature redundancy
8. Applying transformations
9. Scaling the feature matrix
10. Evaluating K=2–8
11. Running fixed-seed stability analysis
12. Applying deterministic K selection
13. Training the final K-Means model
14. Profiling the resulting segments
15. Generating analytical artifacts
16. Persisting the model and transformation metadata

The dashboard consumes persisted outputs rather than rebuilding the complete ML workflow during every page interaction.

---

## Model Persistence & Inference

The trained clustering system persists the artifacts required for reproducible inference.

The inference workflow uses:

```text
Input RFM Profile
      ↓
Schema Validation
      ↓
Constraint Validation
      ↓
Persisted log Transformation
      ↓
Persisted StandardScaler
      ↓
Persisted K-Means Model
      ↓
Segment Mapping
      ↓
Predicted Segment
```

Input validation includes:

* Recency ≥ 0
* Frequency ≥ 1
* Monetary > 0
* NaN rejection
* Infinite-value rejection
* Feature-schema validation
* Out-of-distribution warnings based on persisted training ranges

No retraining occurs during inference.

---

## Testing & Code Quality

The project includes an automated test suite designed to validate core data-processing, feature-engineering, clustering, profiling, and inference behavior.

Tests use synthetic fixtures where appropriate so that CI does not depend on the raw UCI dataset.

### Automated Tests

```bash
python -m pytest -q
```

### Ruff

```bash
python -m ruff check .
```

### Black

```bash
python -m black --check .
```

### Continuous Integration

GitHub Actions validates the project across:

```text
Python 3.11
Python 3.12
```

The CI workflow checks code quality and automated tests.

---

## Project Achievements

* Built an end-to-end customer segmentation pipeline using **541K+ retail transaction records**
* Converted transaction-level data into customer-level behavioral profiles
* Implemented an auditable five-step transaction-cleaning process
* Engineered leakage-controlled RFM features
* Applied skewness-aware feature transformation
* Evaluated feature redundancy before clustering
* Implemented K-Means clustering with documented configuration
* Evaluated K=2 through K=8
* Implemented deterministic evidence-based K selection
* Added fixed five-seed clustering stability analysis
* Used ARI to evaluate consistency across random seeds
* Selected K=3 based on documented evidence and granularity requirements
* Generated automated segment profiles and business interpretations
* Persisted trained model and transformation artifacts
* Implemented schema-validated CLI inference
* Added out-of-distribution warnings for inference inputs
* Developed a seven-page Streamlit retail intelligence application
* Added interactive behavioral and geographic analysis
* Implemented automated testing with Pytest
* Enforced code quality using Ruff and Black
* Configured GitHub Actions CI for Python 3.11 and 3.12
* Documented limitations and avoided unsupported causal claims

---

## Limitations

This project has several important limitations.

* The dataset represents a single retailer and approximately one year of transaction history.
* The customer base is predominantly UK-based, so the segment structure may not generalize to other businesses or markets.
* There is no validated churn outcome in the dataset, so the project does not claim to perform churn prediction.
* A silhouette score of **0.3381** indicates moderate cluster separation rather than strong separation.
* K-Means assumes approximately spherical clusters in the scaled feature space and may not capture every possible customer structure.
* Customer segments describe historical purchasing behavior and should not be interpreted as causal customer types.
* Business recommendations are hypotheses that require controlled experiments or further validation.
* RFM inference requires transaction history; the system does not infer behavioral features from demographic information.
* The selected K is appropriate for the documented dataset and methodology but should be reassessed when the underlying business, time period, or customer population changes.

---

## Reports & Documentation

* **Project Report:** Submitted separately as part of the InternsElite project submission
* **Technical Implementation:** Available in the source code
* **Streamlit Dashboard:** Available through the live application
* **Source Code:** Available in this GitHub repository
* **Dataset Documentation:** Available in `data/README.md`
* **Model Artifacts:** Available in the `models/` directory
* **Generated Analysis:** Available in the `outputs/` directory

---

## References

* [Online Retail Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
* [Python Documentation](https://docs.python.org/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)
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

**InternsElite – AIML Internship

``'
