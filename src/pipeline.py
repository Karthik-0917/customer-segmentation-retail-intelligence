"""Training orchestrator. Run with:  python -m src.pipeline

Flow: load raw -> validate -> clean -> customer aggregation -> RFM +
behavioural features -> diagnostics -> select/transform features -> scale ->
evaluate K=2..8 -> fixed-seed stability -> deterministic multi-criteria K
selection -> train K-Means -> validate -> profile (incl. revenue shares) ->
name segments -> business insights -> PCA projection (visualization only) ->
persist model artifacts + derived analytical outputs for the Streamlit app.
"""

import json
import platform
from datetime import UTC, datetime

import joblib
import pandas as pd
import sklearn
from sklearn.decomposition import PCA

import src
from src.clustering import (
    evaluate_candidate_k,
    select_k,
    stability_by_seed,
    train_kmeans,
    validate_clustering,
)
from src.config import (
    DATASET_PATH,
    FEATURE_SCHEMA_PATH,
    K_MAX,
    K_MIN,
    KMEANS_PATH,
    MODEL_METADATA_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    N_INIT,
    OUTPUTS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    SEGMENT_MAPPING_PATH,
    STABILITY_REFERENCE_SEED,
    STABILITY_SEEDS,
)
from src.data_loader import load_transactions
from src.features import (
    CLUSTERING_FEATURES,
    FEATURE_SCHEMA_VERSION,
    TRANSFORMATION,
    build_customer_features,
    compute_observed_ranges,
    correlation_diagnostics,
    save_feature_schema,
    skewness_diagnostics,
    transform_features,
)
from src.preprocessing import (
    clean_transactions,
    compute_reference_date,
    fit_scaler,
    scale_matrix,
)
from src.profiling import (
    business_insights,
    name_segments,
    population_comparison,
    profile_clusters,
)


def run_pipeline() -> dict:
    """Execute the full training workflow; persist artifacts and outputs."""
    print("[1/9] Loading raw transactions...")
    raw = load_transactions()
    print(f"      {len(raw):,} rows, {raw.shape[1]} columns")

    print("[2/9] Cleaning transactions (documented policy)...")
    transactions, counts = clean_transactions(raw)
    for key, value in counts.items():
        print(f"      {key}: {value:,}")

    reference_date = compute_reference_date(transactions)
    obs_start = transactions["InvoiceDate"].min()
    obs_end = transactions["InvoiceDate"].max()
    total_revenue = float(transactions["Revenue"].sum())
    print(f"      observation: {obs_start:%Y-%m-%d} -> {obs_end:%Y-%m-%d}")
    print(f"      reference date: {reference_date:%Y-%m-%d}")

    print("[3/9] Building customer-level features (RFM + behavioural)...")
    customers = build_customer_features(transactions, reference_date)
    print(
        f"      {len(customers):,} customers, features: "
        f"{[c for c in customers.columns if c != 'CustomerID']}"
    )

    print("[4/9] Feature diagnostics (skewness, Spearman redundancy)...")
    skew = skewness_diagnostics(customers)
    corr = correlation_diagnostics(customers)
    print(skew.to_string())
    print(
        f"      final clustering features: {CLUSTERING_FEATURES} "
        f"(transformation: {TRANSFORMATION}) — see src/features.py rationale"
    )

    print("[5/9] Transforming + scaling final features...")
    transformed = transform_features(customers)
    scaler = fit_scaler(transformed)
    matrix = scale_matrix(scaler, transformed)

    print("[6/9] Evaluating candidate K (elbow + silhouette + fixed-seed stability)...")
    results = evaluate_candidate_k(matrix)
    for r in results:
        print(
            f"      K={r['k']}  inertia={r['inertia']:9.1f}  "
            f"silhouette={r['silhouette']:.4f}  "
            f"sizes {r['min_cluster_size']}..{r['max_cluster_size']}"
        )
        print(
            f"      fixed seed set: {STABILITY_SEEDS} "
            f"(reference seed {STABILITY_REFERENCE_SEED})"
        )
    stability = [stability_by_seed(matrix, k) for k in range(K_MIN, K_MAX + 1)]
    for s in stability:
        print(
            f"      stability K={s['k']}: sil mean={s['sil_mean']:.4f} "
            f"std={s['sil_std']:.4f}  ARI vs seed {s['reference_seed']}: "
            f"mean={s['ari_mean']:.4f} min={s['ari_min']:.4f}"
        )
    best_k, selection = select_k(results, stability, n_customers=len(customers))
    print(f"      selected K={best_k} (deterministic multi-criteria procedure)")
    if selection["business_override"]:
        print(f"      note: {selection['business_override']['reason']}")

    print("[7/9] Training + validating final K-Means...")
    model = train_kmeans(matrix, best_k)
    validation = validate_clustering(model, matrix)
    print(
        f"      inertia={validation['inertia']:.1f}  "
        f"silhouette={validation['silhouette']:.4f}  "
        f"sizes={validation['cluster_sizes']}"
    )

    print("[8/9] Profiling clusters and naming segments...")
    labels = model.labels_
    profiles = profile_clusters(customers, labels)
    names = name_segments(profiles, customers)
    comparison = population_comparison(profiles, customers)
    insights = business_insights(profiles, names, customers)
    for cid, name in sorted(names.items()):
        row = profiles.loc[profiles["cluster"] == cid].iloc[0]
        print(
            f"      Cluster {cid}: {name} ({int(row['count']):,} customers, "
            f"{row['share_pct']}% of base, {row['revenue_share_pct']}% of revenue)"
        )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(matrix)
    pca_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    print(f"      PCA (visualization only) explained variance: {pca_variance}")

    print("[9/9] Persisting model artifacts and analytical outputs...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, KMEANS_PATH)
    observed_ranges = compute_observed_ranges(customers)
    save_feature_schema(FEATURE_SCHEMA_PATH, observed_ranges)

    mapping = {
        "segments": {str(k): v for k, v in names.items()},
        "profiles": profiles.to_dict(orient="records"),
        "n_clusters": best_k,
    }
    SEGMENT_MAPPING_PATH.write_text(json.dumps(mapping, indent=2))

    metadata = {
        "model_version": MODEL_VERSION,
        "app_version": src.__version__,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "UCI Online Retail (id 352)",
        "dataset_file": DATASET_PATH.name,
        "observation_start": f"{obs_start:%Y-%m-%d}",
        "observation_end": f"{obs_end:%Y-%m-%d}",
        "reference_date": f"{reference_date:%Y-%m-%d}",
        "features": CLUSTERING_FEATURES,
        "transformation": TRANSFORMATION,
        "scaler": "StandardScaler",
        "algorithm": "KMeans",
        "selected_k": best_k,
        "k_selection": {
            "procedure": selection["procedure"],
            "parameters": selection["parameters"],
            "quantitative_best": selection["quantitative_best"],
            "business_override": selection["business_override"],
            "explanation": selection["explanation"],
        },
        "stability_seeds": STABILITY_SEEDS,
        "stability_reference_seed": STABILITY_REFERENCE_SEED,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "validation": validation,
        "pca_explained_variance_ratio": pca_variance,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "artifacts": [
            p.name
            for p in (
                SCALER_PATH,
                KMEANS_PATH,
                SEGMENT_MAPPING_PATH,
                FEATURE_SCHEMA_PATH,
                MODEL_METADATA_PATH,
            )
        ],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    # Derived analytical outputs for Streamlit (no raw transaction rows).
    segments_df = customers.copy()
    segments_df["cluster"] = labels
    segments_df["segment"] = segments_df["cluster"].map(names)
    segments_df["PC1"] = pca_coords[:, 0]
    segments_df["PC2"] = pca_coords[:, 1]
    segments_df.to_parquet(OUTPUTS_DIR / "customer_segments.parquet", index=False)

    profiles.assign(segment=profiles["cluster"].map(names)).to_csv(
        OUTPUTS_DIR / "segment_profiles.csv", index=False
    )

    stab_by_k = {s["k"]: s for s in stability}
    sel_by_k = {row["k"]: row for row in selection["table"]}
    eval_rows = []
    for r in results:
        s = stab_by_k[r["k"]]
        t = sel_by_k[r["k"]]
        eval_rows.append(
            {
                **r,
                "min_share": round(t["min_share"], 4),
                "sil_mean_seeds": round(s["sil_mean"], 4),
                "sil_std_seeds": round(s["sil_std"], 4),
                "ari_mean_vs_seed42": round(s["ari_mean"], 4),
                "score": t["score"],
                "status": t["status"],
            }
        )
    pd.DataFrame(eval_rows).to_csv(OUTPUTS_DIR / "model_evaluation.csv", index=False)

    corr.round(4).to_csv(OUTPUTS_DIR / "correlation_matrix.csv")
    skew.to_csv(OUTPUTS_DIR / "feature_summary.csv")
    comparison.to_csv(OUTPUTS_DIR / "population_comparison.csv", index=False)
    (OUTPUTS_DIR / "stability.json").write_text(json.dumps(stability, indent=2))
    (OUTPUTS_DIR / "business_insights.json").write_text(json.dumps(insights, indent=2))

    monthly = (
        transactions.set_index("InvoiceDate")
        .resample("ME")
        .agg(orders=("InvoiceNo", "nunique"), revenue=("Revenue", "sum"))
        .reset_index()
    )
    monthly["InvoiceDate"] = monthly["InvoiceDate"].dt.strftime("%Y-%m")
    monthly.to_csv(OUTPUTS_DIR / "monthly_activity.csv", index=False)

    country = (
        transactions.groupby("Country", observed=True)
        .agg(customers=("CustomerID", "nunique"), revenue=("Revenue", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    country.to_csv(OUTPUTS_DIR / "country_summary.csv", index=False)

    summary = {
        "total_customers": int(len(customers)),
        "valid_orders": counts["valid_orders"],
        "valid_rows": counts["valid_rows"],
        "total_revenue": round(total_revenue, 2),
        "avg_order_value": round(total_revenue / counts["valid_orders"], 2),
        "observation_start": f"{obs_start:%Y-%m-%d}",
        "observation_end": f"{obs_end:%Y-%m-%d}",
        "reference_date": f"{reference_date:%Y-%m-%d}",
        "selected_k": best_k,
        "quantitative_best_k": selection["quantitative_best"],
        "k_selection_reason": selection["explanation"],
        "silhouette": validation["silhouette"],
        "inertia": validation["inertia"],
        "cluster_sizes": {str(k): v for k, v in validation["cluster_sizes"].items()},
        "segments": {str(k): v for k, v in names.items()},
        "pca_explained_variance_ratio": pca_variance,
        "cleaning_counts": counts,
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUTS_DIR / "data_quality.json").write_text(json.dumps(counts, indent=2))
    print(f"      models/: {', '.join(metadata['artifacts'])}")
    print(f"      outputs/: {len(list(OUTPUTS_DIR.iterdir()))} analytical artifacts")

    return {"selected_k": best_k, "validation": validation, "summary": summary}


if __name__ == "__main__":
    run_pipeline()
