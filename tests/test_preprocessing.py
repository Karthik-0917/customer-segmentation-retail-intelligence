"""Cleaning policy, transaction definition, reference date, scaling."""

import numpy as np
import pandas as pd

from src.features import transform_features
from src.preprocessing import (
    clean_transactions,
    compute_reference_date,
    fit_scaler,
    flag_cancellations,
    scale_matrix,
)


def test_cleaning_counts_every_removal(raw_transactions):
    cleaned, counts = clean_transactions(raw_transactions)
    assert counts["rows_loaded"] == len(raw_transactions)
    assert counts["duplicate_rows_removed"] == 1
    assert counts["missing_customer_id_removed"] == 2
    assert counts["cancellation_rows_removed"] == 1
    assert counts["non_positive_qty_or_price_removed"] == 2
    assert counts["valid_rows"] == len(cleaned)
    removed = sum(v for k, v in counts.items() if k.endswith("_removed"))
    assert counts["valid_rows"] == counts["rows_loaded"] - removed


def test_cancellation_detection(raw_transactions):
    mask = flag_cancellations(raw_transactions)
    assert mask.sum() == 1
    assert raw_transactions.loc[mask, "InvoiceNo"].iloc[0] == "C99999"


def test_clean_data_has_positive_revenue_only(raw_transactions):
    cleaned, _ = clean_transactions(raw_transactions)
    assert (cleaned["Revenue"] > 0).all()
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()
    assert cleaned["CustomerID"].notna().all()


def test_order_definition_counts_unique_invoices(raw_transactions):
    cleaned, counts = clean_transactions(raw_transactions)
    assert counts["valid_orders"] == cleaned["InvoiceNo"].nunique()
    # multi-row invoices must not inflate the order count
    assert counts["valid_orders"] < counts["valid_rows"]


def test_reference_date_is_max_date_plus_one_day(raw_transactions):
    cleaned, _ = clean_transactions(raw_transactions)
    reference = compute_reference_date(cleaned)
    assert reference == cleaned["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    # deterministic across calls
    assert reference == compute_reference_date(cleaned)


def test_scaler_round_trip_is_deterministic(clean_customers):
    matrix = transform_features(clean_customers)
    scaler = fit_scaler(matrix)
    scaled = scale_matrix(scaler, matrix)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
    assert np.array_equal(scaled, scale_matrix(scaler, matrix))
