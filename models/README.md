# Model Artifacts

All artifacts in this directory are **generated outputs of the training
pipeline** — never manually created or edited. Recreate them any time with:

```bash
python -m src.pipeline
```

(requires `data/Online_Retail.xlsx`; see `data/README.md`).

## Artifacts

| Artifact               | Contents                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `scaler.joblib`        | `StandardScaler` fitted on the log1p-transformed final feature matrix (Recency, Frequency, Monetary) |
| `kmeans.joblib`        | Final trained `KMeans` model (`random_state=42`, `n_init=10`; K recorded in metadata)             |
| `segment_mapping.json` | `cluster_id → segment name` plus full per-cluster profiles (counts, shares, means, medians)       |
| `feature_schema.json`  | Training/inference contract: feature names, order, `log1p` transformation, dtypes, scaler type, missing-value policy (`reject`), and feature definitions |
| `model_metadata.json`  | Model version, training timestamp, dataset identifier, observation period, reference date, features, transformation, algorithm, selected K + selection reasoning, `random_state`, `n_init`, validation metrics, and key library versions |

## Pipeline that produces them

`python -m src.pipeline` performs: data loading → schema validation →
documented cleaning/cancellation policy → customer-level aggregation → RFM +
behavioural features → skewness & Spearman-redundancy diagnostics → final
feature selection (Recency, Frequency, Monetary) → `log1p` transform →
`StandardScaler` → K=2..8 evaluation (inertia + silhouette) → seed-stability
diagnostics (ARI) → evidence-based K selection → K-Means training →
validation → cluster profiling → data-driven segment naming → business
insights → persistence of these artifacts plus the analytical outputs in
`outputs/`.

## Feature order and transformations (must never drift)

```
0: Recency    — log1p
1: Frequency  — log1p
2: Monetary   — log1p
```

then `StandardScaler`. `feature_schema.json` is the machine-readable source
of truth; `src/inference.py` validates every prediction against it and
rejects mismatches instead of guessing.

## Inference behaviour

`src/inference.py` and the Streamlit predictor **load** these artifacts and
apply them exactly as at training time. Nothing is retrained, refitted,
reordered, or silently imputed at inference. Because the pipeline is
deterministic (fixed `random_state`, deterministic reference date and
cleaning), re-running it on the same dataset reproduces the same artifacts.
