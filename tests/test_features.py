"""RFM, behavioural features, diagnostics, schema, observed ranges."""

import numpy as np
import pytest

from src.features import (
    CANDIDATE_FEATURES,
    CLUSTERING_FEATURES,
    FEATURE_SCHEMA_VERSION,
    build_customer_features,
    build_feature_schema,
    compute_observed_ranges,
    correlation_diagnostics,
    load_feature_schema,
    save_feature_schema,
    skewness_diagnostics,
    transform_features,
)
from src.preprocessing import clean_transactions, compute_reference_date


def test_one_row_per_customer(raw_transactions):
    cleaned, counts = clean_transactions(raw_transactions)
    customers = build_customer_features(cleaned, compute_reference_date(cleaned))
    assert len(customers) == counts["customers"]
    assert customers["CustomerID"].is_unique


def test_rfm_definitions(raw_transactions):
    cleaned, _ = clean_transactions(raw_transactions)
    reference = compute_reference_date(cleaned)
    customers = build_customer_features(cleaned, reference)
    cid = customers["CustomerID"].iloc[0]
    subset = cleaned[cleaned["CustomerID"] == cid]
    row = customers[customers["CustomerID"] == cid].iloc[0]
    assert row["Frequency"] == subset["InvoiceNo"].nunique()
    assert row["Monetary"] == pytest.approx(subset["Revenue"].sum())
    assert row["Recency"] == (reference - subset["InvoiceDate"].max()).days
    assert row["AvgOrderValue"] == pytest.approx(row["Monetary"] / row["Frequency"])


def test_recency_never_negative_no_future_leakage(clean_customers):
    # 0 is legal (purchase hours before the reference date); negative would
    # mean future information leaked into the feature.
    assert (clean_customers["Recency"] >= 0).all()


def test_transform_is_log1p_in_schema_order(clean_customers):
    transformed = transform_features(clean_customers)
    assert list(transformed.columns) == CLUSTERING_FEATURES
    expected = np.log1p(clean_customers[CLUSTERING_FEATURES].astype(float))
    assert np.allclose(transformed.values, expected.values)


def test_transform_rejects_negative_values(clean_customers):
    bad = clean_customers.copy()
    bad.loc[bad.index[0], "Recency"] = -3
    with pytest.raises(ValueError, match="non-negative"):
        transform_features(bad)


def test_diagnostics_cover_candidates(clean_customers):
    corr = correlation_diagnostics(clean_customers)
    assert list(corr.columns) == CANDIDATE_FEATURES
    assert np.allclose(np.diag(corr), 1.0)
    skew = skewness_diagnostics(clean_customers)
    assert set(skew.index) == set(CANDIDATE_FEATURES)


def test_schema_version_and_observed_ranges(clean_customers):
    ranges = compute_observed_ranges(clean_customers)
    schema = build_feature_schema(ranges)
    assert schema["schema_version"] == FEATURE_SCHEMA_VERSION
    for feature in schema["features"]:
        lo, hi = feature["observed_range"]
        assert lo <= hi
        assert lo == float(clean_customers[feature["name"]].min())


def test_schema_round_trip(tmp_path, clean_customers):
    ranges = compute_observed_ranges(clean_customers)
    schema = build_feature_schema(ranges)
    assert [f["name"] for f in schema["features"]] == CLUSTERING_FEATURES
    assert schema["feature_count"] == len(CLUSTERING_FEATURES)
    path = tmp_path / "feature_schema.json"
    save_feature_schema(path, ranges)
    loaded = load_feature_schema(path)
    assert loaded == schema


def test_customer_id_never_a_feature():
    assert "CustomerID" not in CLUSTERING_FEATURES
    assert "CustomerID" not in CANDIDATE_FEATURES
