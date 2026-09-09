"""K-Means training, candidate-K evaluation, fixed-seed stability, and the
deterministic multi-criteria K-selection procedure."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import (
    K_MAX,
    K_MIN,
    K_SELECTION,
    N_INIT,
    RANDOM_STATE,
    STABILITY_REFERENCE_SEED,
    STABILITY_SEEDS,
)

K_SELECTION_PROCEDURE = (
    "Deterministic multi-criteria K selection: (1) evaluate every K in the "
    "configured range, recording inertia, silhouette, cluster sizes, and "
    "fixed-seed stability (per-seed silhouette mean/std and ARI versus the "
    "seed-42 reference). (2) Sanity screens: reject K whose smallest cluster "
    "covers less than the configured share of customers, or whose mean ARI "
    "across the fixed seed set falls below the configured robustness floor. "
    "(3) Evidence score: equal-weighted mean of min-max-normalised "
    "silhouette, mean ARI, and smallest-cluster share across eligible "
    "candidates. (4) Deterministic ranking: score desc, then mean ARI desc, "
    "then smallest-cluster share desc, then lower K. (5) Documented business "
    "granularity layer: if the quantitative leader has fewer than the "
    "configured minimum number of segments required for differentiated "
    "strategy, the top-ranked candidate meeting that minimum is selected and "
    "the tradeoff is recorded explicitly."
)


def make_kmeans(n_clusters: int, random_state: int = RANDOM_STATE) -> KMeans:
    """Explicitly configured KMeans (no reliance on ambiguous defaults)."""
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=N_INIT)


def evaluate_candidate_k(
    matrix: np.ndarray, k_values: range | None = None
) -> list[dict[str, float]]:
    """Fit K-Means for each K; record inertia, silhouette, and size balance."""
    if k_values is None:
        k_values = range(K_MIN, K_MAX + 1)
    results = []
    for k in k_values:
        model = make_kmeans(k)
        labels = model.fit_predict(matrix)
        sizes = np.bincount(labels)
        results.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(matrix, labels)),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            }
        )
    return results


def stability_by_seed(
    matrix: np.ndarray,
    k: int,
    seeds: list[int] | None = None,
    reference_seed: int | None = None,
) -> dict:
    """Fixed-seed stability diagnostics for one candidate K.

    The SAME fixed seed set is used for every candidate K. Reports per-seed
    silhouette (mean and standard deviation across seeds) and the Adjusted
    Rand Index of each non-reference seed's partition versus the
    deterministic reference run (seed 42 by default). The reference is a
    deterministic anchor, not ground truth.
    """
    if seeds is None:
        seeds = STABILITY_SEEDS
    if reference_seed is None:
        reference_seed = STABILITY_REFERENCE_SEED
    labels = {int(s): make_kmeans(k, random_state=int(s)).fit_predict(matrix) for s in seeds}
    sils = {s: float(silhouette_score(matrix, lab)) for s, lab in labels.items()}
    if reference_seed in labels:
        reference = labels[reference_seed]
    else:
        reference = make_kmeans(k, random_state=reference_seed).fit_predict(matrix)
    ari = {
        s: float(adjusted_rand_score(reference, lab))
        for s, lab in labels.items()
        if s != reference_seed
    }
    sil_values = list(sils.values())
    ari_values = list(ari.values())
    return {
        "k": int(k),
        "seeds": [int(s) for s in seeds],
        "reference_seed": int(reference_seed),
        "silhouette_by_seed": sils,
        "sil_mean": float(np.mean(sil_values)),
        "sil_std": float(np.std(sil_values)),
        "ari_vs_reference": ari,
        "ari_scores": ari_values,
        "ari_mean": float(np.mean(ari_values)) if ari_values else 1.0,
        "ari_min": float(np.min(ari_values)) if ari_values else 1.0,
    }


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def select_k(
    results: list[dict],
    stability: list[dict],
    n_customers: int,
    min_cluster_share: float | None = None,
    min_ari: float | None = None,
    min_business_k: int | None = None,
) -> tuple[int, dict]:
    """Apply the documented deterministic K-selection procedure.

    Returns (selected_k, selection_report). The report contains the full
    per-K decision table, the quantitative leader, any recorded business
    override, the generated explanation text, and the procedure description.
    The final K is never hard-coded: it is the output of this procedure
    applied to the actual computed results.
    """
    if min_cluster_share is None:
        min_cluster_share = float(K_SELECTION["min_cluster_share"])
    if min_ari is None:
        min_ari = float(K_SELECTION["min_ari"])
    if min_business_k is None:
        min_business_k = int(K_SELECTION["min_business_k"])

    stab_by_k = {s["k"]: s for s in stability}
    table: list[dict] = []
    for r in results:
        share = r["min_cluster_size"] / n_customers
        ari = float(stab_by_k[r["k"]]["ari_mean"])
        reasons = []
        if share < min_cluster_share:
            reasons.append(
                f"smallest cluster covers {share:.1%} of customers "
                f"(< {min_cluster_share:.0%} sanity floor)"
            )
        if ari < min_ari:
            reasons.append(
                f"mean ARI across fixed seeds {ari:.3f} (< {min_ari} robustness floor)"
            )
        table.append(
            {
                "k": int(r["k"]),
                "silhouette": float(r["silhouette"]),
                "ari_mean": ari,
                "min_share": float(share),
                "eligible": not reasons,
                "rejection_reasons": reasons,
                "score": None,
                "status": "",
            }
        )

    eligible = [row for row in table if row["eligible"]]
    if not eligible:
        raise ValueError(
            "No candidate K passed the sanity screens; inspect the evaluation table."
        )

    sil_norm = _minmax([e["silhouette"] for e in eligible])
    ari_norm = _minmax([e["ari_mean"] for e in eligible])
    share_norm = _minmax([e["min_share"] for e in eligible])
    for row, s, a, b in zip(eligible, sil_norm, ari_norm, share_norm, strict=True):
        row["score"] = round((s + a + b) / 3.0, 6)

    ranked = sorted(
        eligible,
        key=lambda e: (-e["score"], -e["ari_mean"], -e["min_share"], e["k"]),
    )
    quantitative_best = int(ranked[0]["k"])
    selected = quantitative_best
    business_override = None
    if quantitative_best < min_business_k:
        upgraded = [e for e in ranked if e["k"] >= min_business_k]
        if upgraded:
            selected = int(upgraded[0]["k"])
            business_override = {
                "quantitative_best": quantitative_best,
                "min_business_k": min_business_k,
                "reason": (
                    f"K={quantitative_best} leads the quantitative ranking but "
                    f"provides fewer than {min_business_k} segments, which is "
                    "insufficient for differentiated customer strategy (it "
                    "reproduces only a broad active-vs-inactive split). The "
                    "top-ranked candidate meeting the documented minimum was "
                    "selected instead. This qualitative layer is explicit and "
                    "recorded, not a hidden override."
                ),
            }

    for row in table:
        if row["k"] == selected:
            row["status"] = "selected"
        elif business_override and row["k"] == quantitative_best:
            row["status"] = "quantitative leader (below granularity minimum)"
        elif row["eligible"]:
            row["status"] = "eligible"
        else:
            row["status"] = "rejected: " + "; ".join(row["rejection_reasons"])

    explanation = _build_k_explanation(
        table, selected, quantitative_best, business_override, min_business_k
    )
    return selected, {
        "table": table,
        "quantitative_best": quantitative_best,
        "business_override": business_override,
        "explanation": explanation,
        "parameters": {
            "min_cluster_share": min_cluster_share,
            "min_ari": min_ari,
            "min_business_k": min_business_k,
            "score": "equal-weighted mean of min-max-normalised silhouette, "
            "mean ARI, and smallest-cluster share (eligible candidates)",
            "tie_break": "higher mean ARI, then larger smallest-cluster share, then lower K",
        },
        "procedure": K_SELECTION_PROCEDURE,
    }


def _build_k_explanation(
    table: list[dict],
    selected: int,
    quantitative_best: int,
    business_override: dict | None,
    min_business_k: int,
) -> str:
    by_k = {row["k"]: row for row in table}
    sel = by_k[selected]
    parts = []
    if business_override:
        qb = by_k[quantitative_best]
        parts.append(
            f"K={quantitative_best} ranks first on the quantitative evidence score "
            f"(silhouette {qb['silhouette']:.4f}, mean ARI across the fixed seed set "
            f"{qb['ari_mean']:.3f}, smallest-cluster share {qb['min_share']:.1%}), but it "
            f"falls below the documented minimum of {min_business_k} segments required "
            "for differentiated customer strategy: at that coarseness the partition "
            "reproduces only a broad active-versus-inactive split."
        )
        parts.append(
            f"The procedure therefore selects the top-ranked candidate with "
            f"K >= {min_business_k}: K={selected} (evidence score {sel['score']:.3f}, "
            f"silhouette {sel['silhouette']:.4f}, mean ARI {sel['ari_mean']:.3f}, "
            f"smallest cluster {sel['min_share']:.1%} of customers)."
        )
    else:
        parts.append(
            f"K={selected} ranks first on the quantitative evidence score "
            f"({sel['score']:.3f}), with silhouette {sel['silhouette']:.4f}, mean ARI "
            f"across the fixed seed set {sel['ari_mean']:.3f}, and a smallest cluster "
            f"covering {sel['min_share']:.1%} of customers."
        )
    rejected = [row for row in table if not row["eligible"]]
    if rejected:
        parts.append(
            "Rejected by sanity screens: "
            + "; ".join(f"K={r['k']} ({'; '.join(r['rejection_reasons'])})" for r in rejected)
            + "."
        )
    parts.append(
        "The full evidence table for every candidate K, including K=2, is preserved "
        "in outputs/model_evaluation.csv and displayed on the Model Analysis page."
    )
    return " ".join(parts)


def train_kmeans(matrix: np.ndarray, n_clusters: int) -> KMeans:
    """Train the final K-Means model."""
    model = make_kmeans(n_clusters)
    model.fit(matrix)
    return model


def validate_clustering(model: KMeans, matrix: np.ndarray) -> dict:
    """Final-model diagnostics: inertia, silhouette, sizes, centroid gaps.

    Clustering is unsupervised: there is no 'accuracy'. Quality is described
    via cohesion (inertia), separation (silhouette, centroid distances) and
    balance (cluster sizes).
    """
    labels = model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    centers = model.cluster_centers_
    gaps = [
        float(np.linalg.norm(centers[i] - centers[j]))
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    ]
    return {
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(matrix, labels)),
        "cluster_sizes": {int(u): int(c) for u, c in zip(unique, counts, strict=True)},
        "min_centroid_distance": float(np.min(gaps)),
        "mean_centroid_distance": float(np.mean(gaps)),
    }
