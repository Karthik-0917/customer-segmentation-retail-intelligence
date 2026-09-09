"""Plotly chart builders. Segment charts take a segment->color map so the
same segment has the same color on every page."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_MARGIN = dict(l=10, r=10, t=40, b=10)


def segment_distribution(segments: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    counts = segments["segment"].value_counts().reset_index()
    counts.columns = ["segment", "customers"]
    fig = px.bar(
        counts,
        x="customers",
        y="segment",
        orientation="h",
        color="segment",
        color_discrete_map=colors,
        text="customers",
        title="Customers by segment",
    )
    fig.update_layout(showlegend=False, height=320, margin=_MARGIN)
    return fig


def revenue_by_segment(profiles: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    df = profiles.sort_values("revenue")
    fig = px.bar(
        df,
        x="revenue",
        y="segment",
        orientation="h",
        color="segment",
        color_discrete_map=colors,
        text=df["revenue_share_pct"].map(lambda v: f"{v}%"),
        title="Revenue contribution by segment (£)",
    )
    fig.update_layout(showlegend=False, height=320, margin=_MARGIN)
    return fig


def share_comparison(profiles: pd.DataFrame) -> go.Figure:
    long = profiles.melt(
        id_vars=["segment"],
        value_vars=["share_pct", "revenue_share_pct"],
        var_name="measure",
        value_name="pct",
    )
    long["measure"] = long["measure"].map(
        {"share_pct": "Customer share", "revenue_share_pct": "Revenue share"}
    )
    fig = px.bar(
        long,
        x="pct",
        y="segment",
        color="measure",
        barmode="group",
        orientation="h",
        title="Customer share vs revenue share (%)",
        color_discrete_map={"Customer share": "#636EFA", "Revenue share": "#EF553B"},
    )
    fig.update_layout(height=360, margin=_MARGIN, legend_title=None)
    return fig


def cluster_size(profiles: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    df = profiles.sort_values("count")
    fig = px.bar(
        df,
        x="count",
        y="segment",
        orientation="h",
        color="segment",
        color_discrete_map=colors,
        text=[f"{c:,} ({p}%)" for c, p in zip(df["count"], df["share_pct"], strict=True)],
        title="Cluster sizes (count and % of customer base)",
    )
    fig.update_layout(showlegend=False, height=320, margin=_MARGIN)
    return fig


def monthly_trend(monthly: pd.DataFrame, column: str, title: str) -> go.Figure:
    fig = px.line(monthly, x="InvoiceDate", y=column, markers=True, title=title)
    fig.update_layout(height=320, margin=_MARGIN)
    return fig


def country_bar(df: pd.DataFrame, value_col: str, title: str, top_n: int = 12) -> go.Figure:
    top = df.nlargest(top_n, value_col).sort_values(value_col)
    total = df[value_col].sum()
    pct = (100 * top[value_col] / total).round(1)
    fig = px.bar(
        top,
        x=value_col,
        y="Country",
        orientation="h",
        title=title,
        text=[f"{v:,.0f} ({p}%)" for v, p in zip(top[value_col], pct, strict=True)],
    )
    fig.update_layout(height=400, margin=_MARGIN)
    return fig


def elbow_curve(evaluation: pd.DataFrame, selected_k: int) -> go.Figure:
    fig = px.line(
        evaluation,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow Method (inertia vs K, selected K highlighted)",
    )
    sel = evaluation[evaluation["k"] == selected_k]
    if not sel.empty:
        fig.add_trace(
            go.Scatter(
                x=sel["k"],
                y=sel["inertia"],
                mode="markers",
                marker=dict(size=14, color="#EF553B", symbol="circle-open-dot"),
                name=f"Selected K={selected_k}",
            )
        )
    fig.update_layout(height=340, margin=_MARGIN)
    return fig


def silhouette_bars(evaluation: pd.DataFrame, selected_k: int) -> go.Figure:
    colors = ["#EF553B" if k == selected_k else "#636EFA" for k in evaluation["k"]]
    fig = go.Figure(go.Bar(x=evaluation["k"], y=evaluation["silhouette"], marker_color=colors))
    fig.update_layout(
        title="Silhouette score by K (selected K highlighted; K=2 shown, not hidden)",
        xaxis_title="K",
        yaxis_title="Silhouette",
        height=340,
        margin=_MARGIN,
    )
    return fig


def pca_scatter(segments: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    fig = px.scatter(
        segments,
        x="PC1",
        y="PC2",
        color="segment",
        color_discrete_map=colors,
        opacity=0.55,
        hover_data=["Recency", "Frequency", "Monetary"],
        title="Customer clusters — PCA 2D projection (visualization only)",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=460, margin=_MARGIN)
    return fig


def feature_histogram(
    segments: pd.DataFrame, feature: str, log_x: bool, colors: dict[str, str]
) -> go.Figure:
    fig = px.histogram(
        segments,
        x=feature,
        color="segment",
        nbins=60,
        log_x=log_x,
        barmode="overlay",
        opacity=0.6,
        color_discrete_map=colors,
    )
    fig.update_layout(height=380, margin=_MARGIN)
    return fig


def feature_box_by_segment(
    segments: pd.DataFrame, feature: str, log_y: bool, colors: dict[str, str]
) -> go.Figure:
    fig = px.box(
        segments,
        x="segment",
        y=feature,
        color="segment",
        log_y=log_y,
        color_discrete_map=colors,
        category_orders={"segment": sorted(segments["segment"].unique())},
    )
    fig.update_layout(showlegend=False, height=400, margin=_MARGIN)
    return fig


def scatter_explore(
    segments: pd.DataFrame,
    x: str,
    y: str,
    colors: dict[str, str],
    log_x: bool,
    log_y: bool,
) -> go.Figure:
    fig = px.scatter(
        segments,
        x=x,
        y=y,
        color="segment",
        color_discrete_map=colors,
        opacity=0.55,
        log_x=log_x,
        log_y=log_y,
        hover_data=["Recency", "Frequency", "Monetary", "AvgOrderValue"],
        title=f"{y} vs {x} by segment",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=460, margin=_MARGIN)
    return fig


def profile_heatmap(comparison: pd.DataFrame, mapping: dict) -> go.Figure:
    cols = [c for c in comparison.columns if c.endswith("_vs_population")]
    z = comparison[cols].values
    labels = [
        mapping["segments"].get(str(int(c)), f"Cluster {int(c)}")
        for c in comparison["cluster"]
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[c.replace("_vs_population", "") for c in cols],
            y=labels,
            colorscale="RdBu_r",
            zmid=1.0,
            text=[[f"{v:.2f}x" for v in row] for row in z],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Segment medians relative to population median (1.0× = population median)",
        height=380,
        margin=_MARGIN,
    )
    return fig
