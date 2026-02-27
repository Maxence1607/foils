# app.py
# Streamlit app: Foil selection dashboard (Paprec / Vendée Globe)
#
# Features:
# - Load precomputed features/scored data (recommended) OR compute on the fly (optional hook)
# - Foil ranking table + Pareto view
# - Foil detail: distributions, top runs, bad/good tails
# - Run detail: timeseries plots + ACF/FFT (if columns exist) + quick KPIs
#
# Requirements:
#   pip install streamlit pandas numpy altair pyarrow
# Run:
#   streamlit run app.py
#
# Expected files (you can change paths in sidebar):
#   data/outputs/features.parquet
#   data/outputs/scored.parquet (optional; otherwise computed from features)
#
# Notes:
# - This app assumes your scoring pipeline already produced features_df (1 row per run)
# - It can compute scored_df using "axes" scoring (conceptually orthogonal)
# - For run-level timeseries, it reads the CSV referenced in "csv_path" column

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# -----------------------------
# Scoring (axes-based) utilities
# -----------------------------
def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True, method="average")
    return pct if higher_is_better else (1.0 - pct)


def build_run_scores_axes(
    features_df: pd.DataFrame,
    id_cols=("foil_id", "script_id"),
    axes: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    axis_weights: Optional[Dict[str, float]] = None,
    score_scale: float = 100.0,
) -> pd.DataFrame:
    df = features_df.copy()

    if axes is None:
        axes = {
            # Conceptual axes (not mathematically guaranteed orthogonal, but designed to avoid redundancy)
            "performance": [
                {"col": "Boat.Speed_kts__mean", "higher_is_better": True},
            ],
            "short_term_variability": [
                {"col": "Boat.Speed_kts__std", "higher_is_better": False},
                {"col": "Boat.Speed_kts__max_drawdown_pct", "higher_is_better": False},
            ],
            "slow_dynamics": [
                {"col": "Boat.Speed_kts__acf__integral_timescale_seconds", "higher_is_better": False},
                {"col": "Boat.Speed_kts__stationarity__mean_drift_ratio", "higher_is_better": False},
            ],
            "control_effort": [
                {"col": "Boat.Helm__abs_mean", "higher_is_better": False},
                {"col": "Boat.Leeway__abs_mean", "higher_is_better": False},
            ],
        }

    if axis_weights is None:
        axis_weights = {
            "performance": 0.40,
            "short_term_variability": 0.25,
            "slow_dynamics": 0.15,
            "control_effort": 0.20,
        }

    aw = pd.Series(axis_weights, dtype=float)
    aw = aw[aw > 0]
    aw = aw / aw.sum()

    used_metric_cols: List[str] = []

    # Compute metric scores
    for axis_name, metrics in axes.items():
        for spec in metrics:
            col = spec["col"]
            if col not in df.columns:
                continue
            score_col = f"{col}__score"
            if score_col not in df.columns:
                df[score_col] = percentile_score(df[col], spec["higher_is_better"])
            used_metric_cols.append(col)

    # Axis scores: row-wise mean of valid metric scores in the axis
    for axis_name, metrics in axes.items():
        cols = [m["col"] for m in metrics if m["col"] in df.columns]
        score_cols = [f"{c}__score" for c in cols]
        if not score_cols:
            df[f"axis__{axis_name}__score"] = np.nan
            continue

        S = df[score_cols].to_numpy(dtype=float)
        valid = np.isfinite(S)
        denom = valid.sum(axis=1)
        axis_score = np.where(denom > 0, np.nansum(S, axis=1) / denom, np.nan)
        df[f"axis__{axis_name}__score"] = axis_score

    # Global score: weighted mean of axis scores (per-row renormalization if missing axes)
    A = df[[f"axis__{k}__score" for k in aw.index if f"axis__{k}__score" in df.columns]].to_numpy(dtype=float)
    used_axes = [k for k in aw.index if f"axis__{k}__score" in df.columns]
    W = aw.loc[used_axes].to_numpy(dtype=float)

    W_mat = np.tile(W, (len(df), 1))
    validA = np.isfinite(A)
    W_eff = np.where(validA, W_mat, 0.0)
    denom = W_eff.sum(axis=1)
    num = np.nansum(A * W_eff, axis=1)
    run_score_01 = np.where(denom > 0, num / denom, np.nan)

    df["run_score"] = run_score_01 * score_scale

    keep = list(id_cols)
    keep += ["csv_path", "script_name"] if "csv_path" in df.columns and "script_name" in df.columns else []
    keep_metrics = sorted(set(used_metric_cols))
    keep += [c for c in keep_metrics if c in df.columns]
    keep += [f"{c}__score" for c in keep_metrics if f"{c}__score" in df.columns]
    keep += [f"axis__{k}__score" for k in used_axes]
    keep += ["run_score"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def rank_foils_from_runs(
    scored_df: pd.DataFrame,
    foil_col: str = "foil_id",
    run_score_col: str = "run_score",
    topk: int = 10,
    good_quantile: float = 0.85,
    bad_quantile: float = 0.30,
) -> Tuple[pd.DataFrame, float, float]:
    df = scored_df.dropna(subset=[run_score_col]).copy()

    good_thr = float(df[run_score_col].quantile(good_quantile))
    bad_thr = float(df[run_score_col].quantile(bad_quantile))

    out = []
    for foil, g in df.groupby(foil_col):
        g_sorted = g.sort_values(run_score_col, ascending=False)
        top = g_sorted.head(topk)

        out.append({
            "foil_id": foil,
            "n_runs": int(len(g_sorted)),
            "best_run_score": float(g_sorted[run_score_col].iloc[0]),
            "top10_mean_score": float(top[run_score_col].mean()) if len(top) else np.nan,
            "top10_std_score": float(top[run_score_col].std(ddof=1)) if len(top) > 1 else 0.0,
            "top10_min_score": float(top[run_score_col].min()) if len(top) else np.nan,
            f"good_run_pct_>=_{good_thr:.0f}": float((g_sorted[run_score_col] >= good_thr).mean() * 100.0),
            f"bad_run_pct_<=_{bad_thr:.0f}": float((g_sorted[run_score_col] <= bad_thr).mean() * 100.0),
        })

    foil_df = pd.DataFrame(out).sort_values(
        by=["top10_mean_score", "best_run_score", "top10_std_score"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return foil_df, good_thr, bad_thr


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, x_higher_better: bool = True, y_higher_better: bool = True) -> pd.Series:
    """
    Returns boolean mask of non-dominated points in df for two objectives.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if not x_higher_better:
        x = -x
    if not y_higher_better:
        y = -y

    n = len(df)
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        # dominated if exists j: xj>=xi and yj>=yi with at least one strict
        for j in range(n):
            if i == j:
                continue
            if (x[j] >= x[i] and y[j] >= y[i]) and (x[j] > x[i] or y[j] > y[i]):
                is_nd[i] = False
                break
    return pd.Series(is_nd, index=df.index)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("Unsupported format. Use .parquet or .csv")


@st.cache_data(show_spinner=False)
def load_run_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(p)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Foil Selection Dashboard", layout="wide")

st.title("Foil Selection Dashboard")
st.caption("Run-level scoring → Foil-level ranking • Performance vs Robustness • Run deep-dive")

with st.sidebar:
    st.header("Data sources")
    default_features = "data/outputs/features.parquet"
    default_scored = "data/outputs/scored.parquet"

    features_path = st.text_input("features file (.parquet/.csv)", default_features)
    scored_path = st.text_input("scored file (optional)", default_scored)

    st.divider()
    st.header("Scoring (axes)")
    use_pre_scored = st.checkbox("Use precomputed scored file if available", value=True)
    topk = st.slider("Top-K runs for foil KPIs", 5, 20, 10, 1)

    good_q = st.slider("Good quantile (global)", 0.70, 0.95, 0.85, 0.01)
    bad_q = st.slider("Bad quantile (global)", 0.05, 0.50, 0.30, 0.01)

    st.caption("Tip: if scores are compressed, choose quantiles (not fixed thresholds).")

# Load features
try:
    features_df = load_df(features_path)
except Exception as e:
    st.error(f"Cannot load features: {e}")
    st.stop()

# Load scored or compute scored
scored_df = None
if use_pre_scored:
    try:
        if Path(scored_path).exists():
            scored_df = load_df(scored_path)
    except Exception:
        scored_df = None

if scored_df is None:
    # Compute scored from features
    scored_df = build_run_scores_axes(features_df)

# Basic sanity
required_cols = {"foil_id", "run_score"}
missing = [c for c in required_cols if c not in scored_df.columns]
if missing:
    st.error(f"Missing required columns in scored_df: {missing}")
    st.stop()

# Foil ranking
foil_rank_df, good_thr, bad_thr = rank_foils_from_runs(
    scored_df,
    topk=topk,
    good_quantile=good_q,
    bad_quantile=bad_q,
)

# Tabs
tab_overview, tab_foil, tab_run = st.tabs(["Overview", "Foil detail", "Run detail"])

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="top")

    with c1:
        st.subheader("Foil ranking (aggregated)")
        st.caption(f"Good threshold = {good_thr:.2f} (q={good_q:.2f}) • Bad threshold = {bad_thr:.2f} (q={bad_q:.2f})")

        st.dataframe(
            foil_rank_df,
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.subheader("Pareto (performance vs robustness)")
        st.caption("x = topK mean score (higher better) • y = topK std (lower better)")
        tmp = foil_rank_df.copy()

        # Pareto on x=top10_mean_score (higher), y=top10_std_score (lower)
        tmp["pareto"] = pareto_frontier(tmp, "top10_mean_score", "top10_std_score", True, False)

        chart = (
            alt.Chart(tmp)
            .mark_circle(size=180)
            .encode(
                x=alt.X("top10_mean_score:Q", title=f"Top{topk} mean run_score"),
                y=alt.Y("top10_std_score:Q", title=f"Top{topk} std (lower = robust)"),
                color=alt.Color("pareto:N", title="Pareto", scale=alt.Scale(domain=[False, True])),
                tooltip=["foil_id", "n_runs", "best_run_score", "top10_mean_score", "top10_std_score", "top10_min_score"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            "**How to read:** points on the Pareto frontier are not strictly dominated (you can’t improve performance without worsening robustness)."
        )

    st.divider()

    st.subheader("Global run_score distribution")
    hist = (
        alt.Chart(scored_df.dropna(subset=["run_score"]))
        .mark_bar()
        .encode(
            x=alt.X("run_score:Q", bin=alt.Bin(maxbins=40), title="run_score"),
            y=alt.Y("count():Q", title="count"),
            tooltip=[alt.Tooltip("count():Q")],
        )
    )
    st.altair_chart(hist, use_container_width=True)


# -----------------------------
# Foil detail
# -----------------------------
with tab_foil:
    st.subheader("Foil deep dive")

    foil_ids = sorted([x for x in scored_df["foil_id"].dropna().astype(str).unique()])
    if not foil_ids:
        st.warning("No foil_id values found.")
        st.stop()

    sel_foil = st.selectbox("Select foil", foil_ids, index=0)
    g = scored_df[scored_df["foil_id"].astype(str) == str(sel_foil)].copy()
    g = g.dropna(subset=["run_score"]).sort_values("run_score", ascending=False)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Runs", int(len(g)))
    with kpi2:
        st.metric("Best run_score", f"{g['run_score'].max():.2f}")
    with kpi3:
        st.metric(f"Top{topk} mean", f"{g.head(topk)['run_score'].mean():.2f}")
    with kpi4:
        st.metric(f"Top{topk} std", f"{g.head(topk)['run_score'].std(ddof=1):.2f}" if len(g.head(topk)) > 1 else "0.00")

    c1, c2 = st.columns([1.1, 0.9], vertical_alignment="top")

    with c1:
        st.markdown("### Run score distribution (this foil)")
        chart = (
            alt.Chart(g)
            .mark_bar()
            .encode(
                x=alt.X("run_score:Q", bin=alt.Bin(maxbins=25), title="run_score"),
                y=alt.Y("count():Q", title="count"),
                tooltip=[alt.Tooltip("count():Q")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Top runs (clickable CSV path)")
        show_cols = [c for c in ["script_id", "run_score", "csv_path"] if c in g.columns]
        st.dataframe(g[show_cols].head(20), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Axis scores (median over runs)")
        axis_cols = [c for c in g.columns if c.startswith("axis__") and c.endswith("__score")]
        if axis_cols:
            axis_median = g[axis_cols].median(numeric_only=True).reset_index()
            axis_median.columns = ["axis", "median_score"]
            axis_median["axis"] = axis_median["axis"].str.replace("axis__", "").str.replace("__score", "")
            axis_chart = (
                alt.Chart(axis_median)
                .mark_bar()
                .encode(
                    x=alt.X("median_score:Q", title="median axis score (0..1)"),
                    y=alt.Y("axis:N", sort="-x", title="axis"),
                    tooltip=["axis", "median_score"],
                )
            )
            st.altair_chart(axis_chart, use_container_width=True)
        else:
            st.info("No axis score columns found (axis__*__score).")

        st.markdown("### Tails: good & bad frequency (global thresholds)")
        good_pct = float((g["run_score"] >= good_thr).mean() * 100)
        bad_pct = float((g["run_score"] <= bad_thr).mean() * 100)
        st.write(f"- Good runs (>= {good_thr:.2f}): **{good_pct:.1f}%**")
        st.write(f"- Bad runs (<= {bad_thr:.2f}): **{bad_pct:.1f}%**")

    st.divider()
    st.markdown("### Parameter grid (if present)")
    param_candidates = [
        "Boat.Aero.Travel", "Boat.Keel.KeelCant", "Boat.Port.FoilRake",
        "Boat.Stbd.FoilRake", "Boat.TWS_kts"
    ]
    existing_params = [p for p in param_candidates if p in g.columns]
    if existing_params:
        # show run_score by parameter combinations
        pivot_cols = existing_params[:3]  # keep it readable
        grp = g.groupby(pivot_cols, dropna=False)["run_score"].agg(["count", "mean", "max"]).reset_index()
        grp = grp.sort_values("max", ascending=False)
        st.dataframe(grp.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("No parameter columns found in features/scored data. (If you want this, include them in features_df.)")


# -----------------------------
# Run detail
# -----------------------------
with tab_run:
    st.subheader("Run deep dive (time series)")
    st.caption("Select a run (script_id) and load its CSV to inspect time series.")

    # Choose foil first for convenience
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        foil_for_run = st.selectbox("Foil", sorted(scored_df["foil_id"].dropna().astype(str).unique()), index=0)
    g2 = scored_df[scored_df["foil_id"].astype(str) == str(foil_for_run)].dropna(subset=["run_score"]).copy()
    g2 = g2.sort_values("run_score", ascending=False)

    with c2:
        run_choices = g2["script_id"].astype(str).tolist() if "script_id" in g2.columns else []
        if not run_choices:
            st.warning("No script_id column found; cannot select runs.")
            st.stop()
        sel_script = st.selectbox("Run (script_id)", run_choices, index=0)

    run_row = g2[g2["script_id"].astype(str) == str(sel_script)].head(1)
    if run_row.empty:
        st.warning("Run not found.")
        st.stop()

    run_score_val = float(run_row["run_score"].iloc[0])
    st.metric("run_score", f"{run_score_val:.2f}")

    # Show axis breakdown if present
    axis_cols = [c for c in run_row.columns if c.startswith("axis__") and c.endswith("__score")]
    if axis_cols:
        axis_vals = run_row[axis_cols].T.reset_index()
        axis_vals.columns = ["axis", "score"]
        axis_vals["axis"] = axis_vals["axis"].str.replace("axis__", "").str.replace("__score", "")
        axis_chart = (
            alt.Chart(axis_vals)
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="axis score (0..1)"),
                y=alt.Y("axis:N", sort="-x"),
                tooltip=["axis", "score"],
            )
        )
        st.altair_chart(axis_chart, use_container_width=True)

    # Load CSV
    if "csv_path" not in run_row.columns or pd.isna(run_row["csv_path"].iloc[0]):
        st.error("No csv_path found for this run in scored_df/features_df.")
        st.stop()

    csv_path = str(run_row["csv_path"].iloc[0])
    st.write(f"CSV: `{csv_path}`")

    try:
        run_df = load_run_csv(csv_path)
    except Exception as e:
        st.error(f"Cannot load run CSV: {e}")
        st.stop()

    # Time column
    time_col = "Time" if "Time" in run_df.columns else None
    if time_col is None:
        run_df["Time"] = np.arange(len(run_df), dtype=float)
        time_col = "Time"

    # Select series to plot
    default_series = [c for c in ["Boat.Speed_kts", "Boat.Helm", "Boat.Trim", "Boat.Leeway", "Boat.TWA"] if c in run_df.columns]
    series = st.multiselect("Series", options=list(run_df.columns), default=default_series)

    # Downsample slider for faster plotting
    ds = st.slider("Downsample (plot every N points)", 1, 20, 2, 1)
    plot_df = run_df.iloc[::ds].copy()

    # Time series plot (Altair)
    if series:
        long = plot_df[[time_col] + series].copy()
        for c in series:
            long[c] = pd.to_numeric(long[c], errors="coerce")
        long = long.melt(id_vars=[time_col], var_name="series", value_name="value").dropna(subset=["value"])

        ts_chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X(f"{time_col}:Q", title="Time"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("series:N"),
                tooltip=[time_col, "series", "value"],
            )
            .interactive()
        )
        st.altair_chart(ts_chart, use_container_width=True)
    else:
        st.info("Select at least one series to plot.")

    st.divider()
    st.markdown("### Quick KPIs (from scored/features)")
    # show a compact subset if present
    kpi_cols = [
        "Boat.Speed_kts__mean",
        "Boat.Speed_kts__std",
        "Boat.Speed_kts__max_drawdown_pct",
        "Boat.Speed_kts__acf__integral_timescale_seconds",
        "Boat.Speed_kts__stationarity__mean_drift_ratio",
        "Boat.Speed_kts__stationarity__std_drift_ratio",
        "Boat.Helm__abs_mean",
        "Boat.Leeway__abs_mean",
    ]
    present_kpis = [c for c in kpi_cols if c in run_row.columns]
    if present_kpis:
        kpi_vals = run_row[present_kpis].T.reset_index()
        kpi_vals.columns = ["metric", "value"]
        st.dataframe(kpi_vals, use_container_width=True, hide_index=True)
    else:
        st.info("No KPI columns found in scored/features for this run.")

    st.divider()
    st.markdown("### (Optional) ACF/FFT plots")
    st.caption(
        "If you exported arrays for ACF/FFT you can plot them here. "
        "In your current pipeline you stored summary metrics, not full arrays. "
        "To plot ACF/FFT, either (a) recompute them in-app, or (b) save arrays per run."
    )

    st.markdown(
        "**Quick option (recommended):** keep run-detail plots as time series + the ACF/FFT summary metrics. "
        "That’s already very convincing in an interview and much faster."
    )