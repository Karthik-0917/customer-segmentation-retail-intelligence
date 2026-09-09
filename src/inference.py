"""Assign a customer's behavioural profile to a learned segment.

Usage:
    python -m src.inference --recency 30 --frequency 5 --monetary 1500

Semantics: this assigns an EXISTING/external customer's already-computed
behavioural profile (Recency/Frequency/Monetary per the documented
definitions) to the nearest learned K-Means segment. It does not derive RFM
from demographics, and it never retrains or refits anything: the persisted
schema, scaler, and model are loaded and applied exactly as at training.
"""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_SCHEMA_PATH,
    KMEANS_PATH,
    MODEL_METADATA_PATH,
    SCALER_PATH,
    SEGMENT_MAPPING_PATH,
)
from src.features import load_feature_schema


def load_artifacts(
    scaler_path: Path = SCALER_PATH,
    kmeans_path: Path = KMEANS_PATH,
    mapping_path: Path = SEGMENT_MAPPING_PATH,
    schema_path: Path = FEATURE_SCHEMA_PATH,
) -> tuple:
    """Load scaler, K-Means model, segment mapping, and feature schema."""
    for path in (scaler_path, kmeans_path, mapping_path, schema_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model artifact not found: '{path}'. "
                "Train the model first with:  python -m src.pipeline"
            )
    scaler = joblib.load(scaler_path)
    model = joblib.load(kmeans_path)
    mapping = json.loads(Path(mapping_path).read_text())
    schema = load_feature_schema(schema_path)
    _check_compatibility(scaler, model, schema)
    return scaler, model, mapping, schema


def _check_compatibility(scaler, model, schema: dict) -> None:
    """Reject artifact/schema mismatches instead of guessing."""
    n = schema["feature_count"]
    if getattr(scaler, "n_features_in_", n) != n:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but the schema "
            f"defines {n}. Re-run: python -m src.pipeline"
        )
    if model.cluster_centers_.shape[1] != n:
        raise ValueError(
            f"K-Means was trained on {model.cluster_centers_.shape[1]} features "
            f"but the schema defines {n}. Re-run: python -m src.pipeline"
        )


def _check_constraint(name: str, value: float) -> str | None:
    """Feature-specific validity rules; returns an error message or None."""
    if name == "Recency" and value < 0:
        return "Recency must be >= 0 (days since the most recent valid purchase)"
    if name == "Frequency" and value < 1:
        return "Frequency must be >= 1 (at least one valid order)"
    if name == "Monetary" and value <= 0:
        return "Monetary must be > 0 (total valid purchase revenue)"
    if name not in {"Recency", "Frequency", "Monetary"} and value < 0:
        return f"Feature '{name}' must be non-negative"
    return None


def validate_inputs(inputs: dict[str, float], schema: dict) -> pd.DataFrame:
    """Validate names, completeness, numeric types, finiteness, and ranges.

    Missing features are rejected (never silently filled); extra features
    are rejected (never silently dropped); order comes from the schema.
    """
    expected = [f["name"] for f in sorted(schema["features"], key=lambda f: f["order"])]
    missing = [name for name in expected if name not in inputs]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    extra = [name for name in inputs if name not in expected]
    if extra:
        raise ValueError(f"Unexpected features: {extra}. Expected exactly: {expected}")
    values = []
    for name in expected:
        value = inputs[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{name}' must be numeric, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite, got {value}")
        problem = _check_constraint(name, value)
        if problem:
            raise ValueError(problem)
        values.append(value)
    return pd.DataFrame([values], columns=expected)


def out_of_distribution_warnings(inputs: dict[str, float], schema: dict) -> list[str]:
    """Warn (not reject) when an input lies outside the observed training range."""
    warnings = []
    for feature in schema["features"]:
        rng = feature.get("observed_range")
        if not rng:
            continue
        name = feature["name"]
        if name not in inputs:
            continue
        value = float(inputs[name])
        if value < rng[0] or value > rng[1]:
            warnings.append(
                f"{name}={value:g} is outside the observed training range "
                f"[{rng[0]:g}, {rng[1]:g}]; the assignment may be less reliable."
            )
    return warnings


def assign_segment(
    inputs: dict[str, float],
    scaler=None,
    model=None,
    mapping: dict | None = None,
    schema: dict | None = None,
) -> dict:
    """Assign a behavioural profile to the nearest learned segment.

    Applies the schema-defined log1p transform and the persisted training
    scaler, then predicts with the persisted K-Means model. No retraining.
    """
    if any(a is None for a in (scaler, model, mapping, schema)):
        scaler, model, mapping, schema = load_artifacts()
    frame = validate_inputs(inputs, schema)
    transformations = {f["name"]: f["transformation"] for f in schema["features"]}
    for col in frame.columns:
        if transformations.get(col) == "log1p":
            frame[col] = np.log1p(frame[col])
    scaled = scaler.transform(frame)
    cluster = int(model.predict(scaled)[0])
    segment_name = mapping["segments"].get(str(cluster), f"Cluster {cluster}")
    profile = next((p for p in mapping.get("profiles", []) if p["cluster"] == cluster), None)
    return {
        "cluster": cluster,
        "segment_name": segment_name,
        "profile": profile,
        "warnings": out_of_distribution_warnings(inputs, schema),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign an existing/external customer's behavioural profile "
            "(Recency/Frequency/Monetary) to a learned segment."
        )
    )
    parser.add_argument(
        "--recency",
        type=float,
        required=True,
        help="Days since the customer's most recent valid purchase (>= 0)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        required=True,
        help="Number of unique valid purchase orders (>= 1)",
    )
    parser.add_argument(
        "--monetary",
        type=float,
        required=True,
        help="Total valid purchase revenue for the customer (> 0)",
    )
    args = parser.parse_args()

    result = assign_segment(
        {"Recency": args.recency, "Frequency": args.frequency, "Monetary": args.monetary}
    )
    print(f"Assigned segment : Cluster {result['cluster']} — {result['segment_name']}")
    if result["profile"]:
        p = result["profile"]
        print(
            f"Segment profile  : {p['count']} customers ({p['share_pct']}%), "
            f"median recency {p['median_Recency']:.0f} d, "
            f"median {p['median_Frequency']:.0f} orders, "
            f"median spend {p['median_Monetary']:,.0f}, "
            f"revenue share {p.get('revenue_share_pct', 'n/a')}%"
        )
    for warning in result["warnings"]:
        print(f"Warning          : {warning}")
    if Path(MODEL_METADATA_PATH).exists():
        meta = json.loads(Path(MODEL_METADATA_PATH).read_text())
        print(
            f"Model            : {meta.get('model_version')} "
            f"(K={meta.get('selected_k')}, schema v{meta.get('feature_schema_version')})"
        )


if __name__ == "__main__":
    main()
