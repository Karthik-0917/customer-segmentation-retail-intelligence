"""K evaluation, training, validation, fixed-seed stability, K selection."""

import numpy as np

from src.clustering import (
    evaluate_candidate_k,
    make_kmeans,
    select_k,
    stability_by_seed,
    train_kmeans,
    validate_clustering,
)
from src.config import N_INIT, RANDOM_STATE, STABILITY_SEEDS
from src.features import transform_features
from src.preprocessing import fit_scaler, scale_matrix


def _matrix(clean_customers):
    transformed = transform_features(clean_customers)
    scaler = fit_scaler(transformed)
    return scale_matrix(scaler, transformed)


def test_kmeans_config_is_explicit():
    model = make_kmeans(3)
    assert model.random_state == RANDOM_STATE
    assert model.n_init == N_INIT


def test_fixed_seed_set_is_configured():
    assert STABILITY_SEEDS == [42, 7, 21, 52, 101]


def test_evaluate_candidate_k_metrics(clean_customers):
    results = evaluate_candidate_k(_matrix(clean_customers), range(2, 5))
    assert [r["k"] for r in results] == [2, 3, 4]
    inertias = [r["inertia"] for r in results]
    assert all(a >= b for a, b in zip(inertias, inertias[1:], strict=False))
    assert all(-1 <= r["silhouette"] <= 1 for r in results)
    assert all(r["min_cluster_size"] >= 1 for r in results)


def test_training_is_reproducible(clean_customers):
    matrix = _matrix(clean_customers)
    m1, m2 = train_kmeans(matrix, 3), train_kmeans(matrix, 3)
    assert np.array_equal(m1.labels_, m2.labels_)
    assert np.allclose(m1.cluster_centers_, m2.cluster_centers_)


def test_validation_report(clean_customers):
    matrix = _matrix(clean_customers)
    model = train_kmeans(matrix, 3)
    report = validate_clustering(model, matrix)
    assert report["inertia"] > 0
    assert -1 <= report["silhouette"] <= 1
    assert sum(report["cluster_sizes"].values()) == len(matrix)
    assert report["min_centroid_distance"] > 0


def test_stability_structure_and_determinism(clean_customers):
    matrix = _matrix(clean_customers)
    s1 = stability_by_seed(matrix, 2, seeds=[42, 7, 21], reference_seed=42)
    s2 = stability_by_seed(matrix, 2, seeds=[42, 7, 21], reference_seed=42)
    assert s1 == s2
    assert s1["seeds"] == [42, 7, 21]
    assert len(s1["ari_scores"]) == 2  # reference seed excluded from ARI
    assert s1["sil_std"] >= 0
    assert -1 <= s1["ari_min"] <= s1["ari_mean"] <= 1


def _demo_inputs():
    results = [
        {
            "k": 2,
            "inertia": 100.0,
            "silhouette": 0.50,
            "min_cluster_size": 40,
            "max_cluster_size": 60,
        },
        {
            "k": 3,
            "inertia": 70.0,
            "silhouette": 0.40,
            "min_cluster_size": 25,
            "max_cluster_size": 45,
        },
        {
            "k": 4,
            "inertia": 55.0,
            "silhouette": 0.35,
            "min_cluster_size": 20,
            "max_cluster_size": 40,
        },
    ]
    stability = [
        {"k": 2, "ari_mean": 0.99},
        {"k": 3, "ari_mean": 0.97},
        {"k": 4, "ari_mean": 0.90},
    ]
    return results, stability


def test_select_k_records_business_override():
    results, stability = _demo_inputs()
    k, report = select_k(
        results,
        stability,
        n_customers=100,
        min_cluster_share=0.02,
        min_ari=0.85,
        min_business_k=3,
    )
    assert report["quantitative_best"] == 2
    assert k == 3
    assert report["business_override"] is not None
    statuses = {row["k"]: row["status"] for row in report["table"]}
    assert statuses[3] == "selected"
    assert "quantitative leader" in statuses[2]
    assert "K=2" in report["explanation"]


def test_select_k_without_business_constraint():
    results, stability = _demo_inputs()
    k, report = select_k(
        results,
        stability,
        n_customers=100,
        min_cluster_share=0.02,
        min_ari=0.85,
        min_business_k=2,
    )
    assert k == 2
    assert report["business_override"] is None


def test_select_k_sanity_rejections():
    results, stability = _demo_inputs()
    results[2]["min_cluster_size"] = 1  # 1% share -> below 2% floor
    stability[1]["ari_mean"] = 0.50  # K=3 unstable
    k, report = select_k(
        results,
        stability,
        n_customers=100,
        min_cluster_share=0.02,
        min_ari=0.85,
        min_business_k=2,
    )
    statuses = {row["k"]: row["status"] for row in report["table"]}
    assert statuses[4].startswith("rejected")
    assert statuses[3].startswith("rejected")
    assert k == 2


def test_select_k_deterministic_tie_break_prefers_lower_k():
    results = [
        {
            "k": 2,
            "inertia": 100.0,
            "silhouette": 0.30,
            "min_cluster_size": 40,
            "max_cluster_size": 60,
        },
        {
            "k": 3,
            "inertia": 70.0,
            "silhouette": 0.40,
            "min_cluster_size": 25,
            "max_cluster_size": 45,
        },
        {
            "k": 4,
            "inertia": 55.0,
            "silhouette": 0.40,
            "min_cluster_size": 25,
            "max_cluster_size": 40,
        },
    ]
    stability = [
        {"k": 2, "ari_mean": 0.90},
        {"k": 3, "ari_mean": 0.95},
        {"k": 4, "ari_mean": 0.95},
    ]
    k, report = select_k(
        results,
        stability,
        n_customers=100,
        min_cluster_share=0.02,
        min_ari=0.85,
        min_business_k=2,
    )
    assert k == 3  # identical evidence for K=3/K=4 -> lower K wins
    k2, _ = select_k(
        results,
        stability,
        n_customers=100,
        min_cluster_share=0.02,
        min_ari=0.85,
        min_business_k=2,
    )
    assert k2 == k  # deterministic across reruns
