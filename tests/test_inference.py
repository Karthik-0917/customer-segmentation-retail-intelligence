"""Persistence round-trip and strict inference contract (incl. OOD warnings)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.clustering import train_kmeans
from src.features import (
    compute_observed_ranges,
    save_feature_schema,
    transform_features,
)
from src.inference import (
    assign_segment,
    load_artifacts,
    out_of_distribution_warnings,
    validate_inputs,
)
from src.preprocessing import fit_scaler, scale_matrix
from src.profiling import name_segments, profile_clusters


@pytest.fixture()
def artifacts(tmp_path, clean_customers):
    """Train on synthetic customers, persist artifacts to a temp dir."""
    transformed = transform_features(clean_customers)
    scaler = fit_scaler(transformed)
    model = train_kmeans(scale_matrix(scaler, transformed), 3)
    profiles = profile_clusters(clean_customers, model.labels_)
    names = name_segments(profiles, clean_customers)

    paths = (
        tmp_path / "scaler.joblib",
        tmp_path / "kmeans.joblib",
        tmp_path / "segment_mapping.json",
        tmp_path / "feature_schema.json",
    )
    joblib.dump(scaler, paths[0])
    joblib.dump(model, paths[1])
    paths[2].write_text(
        json.dumps(
            {
                "segments": {str(k): v for k, v in names.items()},
                "profiles": profiles.to_dict(orient="records"),
                "n_clusters": 3,
            }
        )
    )
    save_feature_schema(paths[3], compute_observed_ranges(clean_customers))
    return paths


def test_missing_artifact_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.pipeline"):
        load_artifacts(tmp_path / "s", tmp_path / "k", tmp_path / "m", tmp_path / "f")


def test_round_trip_and_compatibility(artifacts):
    scaler, model, mapping, schema = load_artifacts(*artifacts)
    assert model.n_clusters == mapping["n_clusters"] == 3
    assert scaler.n_features_in_ == schema["feature_count"]
    assert schema["schema_version"]


def test_assignment_matches_manual_pipeline(artifacts):
    """Inference must exactly reproduce training preprocessing (no refit)."""
    loaded = load_artifacts(*artifacts)
    scaler, model, mapping, schema = loaded
    inputs = {"Recency": 20.0, "Frequency": 4.0, "Monetary": 500.0}
    result = assign_segment(inputs, *loaded)
    manual = pd.DataFrame(
        [[np.log1p(20.0), np.log1p(4.0), np.log1p(500.0)]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    expected = int(model.predict(scaler.transform(manual))[0])
    assert result["cluster"] == expected
    assert result["segment_name"] == mapping["segments"][str(expected)]
    assert result["profile"]["cluster"] == expected


def test_prediction_does_not_retrain_or_mutate(artifacts):
    """Fails if retraining or refitting is ever introduced into inference."""
    loaded = load_artifacts(*artifacts)
    scaler, model, _, _ = loaded
    centers_before = model.cluster_centers_.copy()
    scaler_mean_before = scaler.mean_.copy()
    assign_segment({"Recency": 5, "Frequency": 2, "Monetary": 100}, *loaded)
    assert np.array_equal(model.cluster_centers_, centers_before)
    assert np.array_equal(scaler.mean_, scaler_mean_before)


def test_strict_schema_validation(artifacts):
    loaded = load_artifacts(*artifacts)
    schema = loaded[3]
    with pytest.raises(ValueError, match="Missing required features"):
        validate_inputs({"Recency": 1.0, "Frequency": 2.0}, schema)
    with pytest.raises(ValueError, match="Unexpected features"):
        validate_inputs({"Recency": 1, "Frequency": 2, "Monetary": 3, "Age": 30}, schema)
    with pytest.raises(ValueError, match="numeric"):
        validate_inputs({"Recency": "ten", "Frequency": 2, "Monetary": 3}, schema)
    with pytest.raises(ValueError, match="finite"):
        validate_inputs({"Recency": float("inf"), "Frequency": 2, "Monetary": 3}, schema)


def test_domain_constraints(artifacts):
    loaded = load_artifacts(*artifacts)
    schema = loaded[3]
    with pytest.raises(ValueError, match="Recency"):
        validate_inputs({"Recency": -1, "Frequency": 2, "Monetary": 3}, schema)
    with pytest.raises(ValueError, match="Frequency"):
        validate_inputs({"Recency": 1, "Frequency": 0, "Monetary": 3}, schema)
    with pytest.raises(ValueError, match="Monetary"):
        validate_inputs({"Recency": 1, "Frequency": 2, "Monetary": 0}, schema)
    # Recency of exactly 0 is valid (purchased earlier the same day).
    result = assign_segment({"Recency": 0.0, "Frequency": 1.0, "Monetary": 10.0}, *loaded)
    assert isinstance(result["cluster"], int)


def test_out_of_distribution_warning(artifacts, clean_customers):
    loaded = load_artifacts(*artifacts)
    schema = loaded[3]
    max_monetary = float(clean_customers["Monetary"].max())
    ood = {"Recency": 1.0, "Frequency": 1.0, "Monetary": max_monetary * 100}
    warnings = out_of_distribution_warnings(ood, schema)
    assert any("Monetary" in w for w in warnings)
    result = assign_segment(ood, *loaded)
    assert result["warnings"]
    in_range = {
        "Recency": float(clean_customers["Recency"].median()),
        "Frequency": float(clean_customers["Frequency"].median()),
        "Monetary": float(clean_customers["Monetary"].median()),
    }
    assert out_of_distribution_warnings(in_range, schema) == []


def test_schema_model_mismatch_rejected(artifacts, tmp_path, clean_customers):
    """A scaler trained on a different feature count must be rejected."""
    _, kmeans_p, mapping_p, schema_p = artifacts
    wrong = transform_features(clean_customers).iloc[:, :2]
    bad_scaler = fit_scaler(wrong)
    bad_path = tmp_path / "bad_scaler.joblib"
    joblib.dump(bad_scaler, bad_path)
    with pytest.raises(ValueError, match="Scaler expects"):
        load_artifacts(bad_path, kmeans_p, mapping_p, schema_p)
