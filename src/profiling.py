"""Cluster profiling, data-driven segment naming, business interpretation."""

import numpy as np
import pandas as pd

from src.features import CLUSTERING_FEATURES

PROFILE_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalQuantity",
    "ActiveDays",
]


def profile_clusters(customers: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster counts, shares, revenue contribution, and mean/median
    statistics of behavioural features.

    Medians are reported alongside means because monetary features are
    heavily right-skewed.
    """
    data = customers.copy()
    data["cluster"] = labels
    total = len(data)
    total_revenue = float(data["Monetary"].sum())
    rows = []
    for cluster_id, group in data.groupby("cluster"):
        revenue = float(group["Monetary"].sum())
        row: dict[str, float] = {
            "cluster": int(cluster_id),
            "count": int(len(group)),
            "share_pct": round(100 * len(group) / total, 1),
            "revenue": round(revenue, 2),
            "revenue_share_pct": (
                round(100 * revenue / total_revenue, 1) if total_revenue else 0.0
            ),
        }
        for col in PROFILE_FEATURES:
            row[f"mean_{col}"] = round(float(group[col].mean()), 2)
            row[f"median_{col}"] = round(float(group[col].median()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def population_comparison(profiles: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Ratio of each cluster's median to the overall population median."""
    comparison = profiles[["cluster", "count", "share_pct"]].copy()
    for col in CLUSTERING_FEATURES + ["AvgOrderValue"]:
        overall = float(customers[col].median())
        if overall != 0:
            comparison[f"{col}_vs_population"] = (profiles[f"median_{col}"] / overall).round(2)
    return comparison


def _rfm_level(value: float, population: pd.Series) -> str:
    """Describe a cluster median as Low/Mid/High vs population terciles."""
    if value <= population.quantile(0.33):
        return "Low"
    if value >= population.quantile(0.67):
        return "High"
    return "Mid"


_ACTIVITY = {"Low": "Active", "Mid": "Steady", "High": "Inactive"}


def name_segments(profiles: pd.DataFrame, customers: pd.DataFrame) -> dict[int, str]:
    """Derive segment names from each cluster's actual median RFM profile.

    Naming is tercile-based against the customer population; nothing is
    predetermined. Two dimensions are combined so every name is unique and
    fully data-driven: an activity word from the Recency tercile (Recency is
    inverted: fewer days = more recently active) and a value phrase from the
    Frequency/Monetary terciles.
    """
    names: dict[int, str] = {}
    for _, row in profiles.iterrows():
        recency = _rfm_level(row["median_Recency"], customers["Recency"])
        frequency = _rfm_level(row["median_Frequency"], customers["Frequency"])
        monetary = _rfm_level(row["median_Monetary"], customers["Monetary"])

        if frequency == "High" and monetary == "High":
            value = "High-Value Frequent Buyers"
        elif frequency == "Low" and monetary == "Low":
            value = "Low-Spend Infrequent Customers"
        elif monetary == "High":
            value = "High-Spend Occasional Buyers"
        else:
            value = "Mid-Value Customers"
        name = f"{_ACTIVITY[recency]} {value}"
        # Guarantee uniqueness in the unlikely event of identical profiles.
        if name in names.values():
            name = f"{name} (Cluster {int(row['cluster'])})"
        names[int(row["cluster"])] = name
    return names


def business_insights(
    profiles: pd.DataFrame, names: dict[int, str], customers: pd.DataFrame
) -> list[dict]:
    """Observational, hypothesis-oriented insight blocks per segment.

    Deliberately avoids causal claims and undefined concepts such as churn:
    clustering describes behaviour, it does not prove that any strategy will
    change it, and this dataset contains no validated churn outcome.
    """
    insights = []
    med = {c: float(customers[c].median()) for c in ("Recency", "Frequency", "Monetary")}
    for _, row in profiles.sort_values("cluster").iterrows():
        cid = int(row["cluster"])
        r, f, m = row["median_Recency"], row["median_Frequency"], row["median_Monetary"]
        observed = (
            f"Median recency {r:.0f} days (population {med['Recency']:.0f}), "
            f"median {f:.0f} orders (population {med['Frequency']:.0f}), "
            f"median spend {m:,.0f} (population {med['Monetary']:,.0f}). "
            f"{int(row['count'])} customers ({row['share_pct']}% of the base) "
            f"contributing {row['revenue_share_pct']}% of valid revenue."
        )
        if f >= med["Frequency"] and m >= med["Monetary"] and r <= med["Recency"]:
            strategy = (
                "This segment may be a candidate for loyalty or early-access "
                "programmes; retention experiments could be tested here."
            )
            risk = "Concentration risk: a large revenue share depends on this group."
        elif r > med["Recency"] and m >= med["Monetary"]:
            strategy = (
                "The observed profile suggests previously valuable customers whose "
                "purchases have slowed; a potential re-engagement strategy could be "
                "evaluated through a controlled experiment."
            )
            risk = (
                "Sustained high recency reflects reduced recent purchasing "
                "activity; the trend may warrant monitoring."
            )
        elif r > med["Recency"]:
            strategy = (
                "A potential low-cost reactivation test (e.g. win-back offers) "
                "could be evaluated; heavy investment may not be justified."
            )
            risk = "Low expected return; acquisition economics should be reviewed."
        else:
            strategy = (
                "This segment may warrant nurture campaigns aimed at increasing "
                "purchase frequency; cross-sell tests could be considered."
            )
            risk = "Early-stage relationship: behaviour may still shift materially."
        insights.append(
            {
                "cluster": cid,
                "segment": names[cid],
                "observed_profile": observed,
                "customer_share_pct": float(row["share_pct"]),
                "revenue_share_pct": float(row["revenue_share_pct"]),
                "potential_strategy": strategy,
                "risk_or_opportunity": risk,
                "caution": (
                    "Descriptive clustering only: these groupings do not establish "
                    "that any strategy will cause revenue or retention changes."
                ),
            }
        )
    return insights
