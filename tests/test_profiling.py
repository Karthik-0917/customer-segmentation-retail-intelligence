"""Cluster profiling, naming, revenue shares, and business insights."""

import pytest

from src.clustering import train_kmeans
from src.features import transform_features
from src.preprocessing import fit_scaler, scale_matrix
from src.profiling import (
    business_insights,
    name_segments,
    population_comparison,
    profile_clusters,
)

FORBIDDEN = ("will increase", "will reduce", "guarantees", "proves", "causes", "churn")


def _fit(clean_customers, k=3):
    transformed = transform_features(clean_customers)
    scaler = fit_scaler(transformed)
    model = train_kmeans(scale_matrix(scaler, transformed), k)
    return model.labels_


def test_profiles_cover_all_customers(clean_customers):
    labels = _fit(clean_customers)
    profiles = profile_clusters(clean_customers, labels)
    assert len(profiles) == 3
    assert profiles["count"].sum() == len(clean_customers)
    assert abs(profiles["share_pct"].sum() - 100) < 0.5
    assert "median_Monetary" in profiles.columns
    assert "mean_Recency" in profiles.columns


def test_profiles_include_revenue_contribution(clean_customers):
    labels = _fit(clean_customers)
    profiles = profile_clusters(clean_customers, labels)
    assert profiles["revenue"].sum() == pytest.approx(
        float(clean_customers["Monetary"].sum()), rel=1e-6
    )
    assert abs(profiles["revenue_share_pct"].sum() - 100) < 0.5


def test_population_comparison_ratios(clean_customers):
    labels = _fit(clean_customers)
    profiles = profile_clusters(clean_customers, labels)
    comparison = population_comparison(profiles, clean_customers)
    assert "Monetary_vs_population" in comparison.columns
    assert (comparison["Monetary_vs_population"] > 0).all()


def test_segment_names_unique_and_data_driven(clean_customers):
    labels = _fit(clean_customers)
    profiles = profile_clusters(clean_customers, labels)
    names = name_segments(profiles, clean_customers)
    assert set(names) == {0, 1, 2}
    assert len(set(names.values())) == 3
    assert all(isinstance(n, str) and n for n in names.values())


def test_insights_avoid_causal_and_churn_language(clean_customers):
    labels = _fit(clean_customers)
    profiles = profile_clusters(clean_customers, labels)
    names = name_segments(profiles, clean_customers)
    insights = business_insights(profiles, names, clean_customers)
    assert len(insights) == 3
    for item in insights:
        assert "caution" in item
        assert "revenue_share_pct" in item
        text = " ".join(str(v) for v in item.values()).lower()
        assert not any(phrase in text for phrase in FORBIDDEN)
