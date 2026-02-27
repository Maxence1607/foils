# app.py
# Streamlit dashboard with:
# - Overview (ranking + Pareto + global distributions)
# - Features & Methodology (explanations per feature + per axis)
# - Foil detail (distributions, top runs, tails, parameters)
# - Run detail (timeseries + KPIs)
# - Decision (recommended foils + rationale + trade-offs + sensitivity)
# - Interactive axis weights (recomputes run_score + rankings live)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# =============================
# Scoring utilities (axes)
# =============================
def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True, method="average")
    return pct if higher_is_better else (1.0 - pct)


def compute_metric_scores(df: pd.DataFrame, axes: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Adds per-metric percentile scores: <col>__score (0..1 oriented so higher=better).
    """
    out = df.copy()
    for axis_name, metrics in axes.items():
        for spec in metrics:
            col = spec["col"]
            if col not in out.columns:
                continue
            score_col = f"{col}__score"
            if score_col not in out.columns:
                out[score_col] = percentile_score(out[col], higher_is_better=spec["higher_is_better"])
    return out


def compute_axis_scores(df: pd.DataFrame, axes: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Adds axis scores: axis__<axis>__score as row-wise mean of valid metric scores in that axis.
    """
    out = df.copy()
    for axis_name, metrics in axes.items():
        cols = [m["col"] for m in metrics if m["col"] in out.columns]
        score_cols = [f"{c}__score" for c in cols if f"{c}__score" in out.columns]

        if not score_cols:
            out[f"axis__{axis_name}__score"] = np.nan
            continue

        S = out[score_cols].to_numpy(dtype=float)
        valid = np.isfinite(S)
        denom = valid.sum(axis=1)
        axis_score = np.where(denom > 0, np.nansum(S, axis=1) / denom, np.nan)
        out[f"axis__{axis_name}__score"] = axis_score
    return out


def compute_run_score_from_axes(
    df: pd.DataFrame,
    axis_weights: Dict[str, float],
    score_scale: float = 100.0,
) -> pd.DataFrame:
    """
    Computes run_score from axis__*__score using weights (per-row renorm if missing).
    """
    out = df.copy()
    aw = pd.Series(axis_weights, dtype=float)
    aw = aw[aw > 0]
    aw = aw / aw.sum()

    used_axes = [a for a in aw.index if f"axis__{a}__score" in out.columns]
    if not used_axes:
        out["run_score"] = np.nan
        return out

    A = out[[f"axis__{a}__score" for a in used_axes]].to_numpy(dtype=float)
    W = aw.loc[used_axes].to_numpy(dtype=float)

    W_mat = np.tile(W, (len(out), 1))
    validA = np.isfinite(A)
    W_eff = np.where(validA, W_mat, 0.0)
    denom = W_eff.sum(axis=1)
    num = np.nansum(A * W_eff, axis=1)
    run_score_01 = np.where(denom > 0, num / denom, np.nan)
    out["run_score"] = run_score_01 * score_scale
    return out


def build_scored_df(
    features_df: pd.DataFrame,
    axes: Dict[str, List[Dict[str, Any]]],
    axis_weights: Dict[str, float],
    score_scale: float = 100.0,
) -> pd.DataFrame:
    df = features_df.copy()
    df = compute_metric_scores(df, axes)
    df = compute_axis_scores(df, axes)
    df = compute_run_score_from_axes(df, axis_weights, score_scale=score_scale)
    return df


def rank_foils_from_runs(
    scored_df: pd.DataFrame,
    foil_col: str = "foil_id",
    run_score_col: str = "run_score",
    topk: int = 10,
    good_quantile: float = 0.85,
    bad_quantile: float = 0.30,
    plateau_frac: float = 0.95,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Aggregate run-level scores into foil-level KPIs.

    Philosophy (matches the "many simulations per foil" setting):
      - Use Top-K mean as a proxy for *potential* (best achievable with tuning).
      - Use Top-K std / IQR as proxies for *robustness* (how sensitive performance is).
      - Use tails (good% / bad%) for risk.
      - Use plateau% (runs within plateau_frac of best) for "how wide is the sweet spot".
    """
    df = scored_df.dropna(subset=[foil_col, run_score_col]).copy()
    df[foil_col] = df[foil_col].astype(str)

    good_thr = float(df[run_score_col].quantile(good_quantile))
    bad_thr = float(df[run_score_col].quantile(bad_quantile))

    out: List[Dict[str, Any]] = []
    for foil, g in df.groupby(foil_col):
        g_sorted = g.sort_values(run_score_col, ascending=False)
        top = g_sorted.head(topk)

        best = float(g_sorted[run_score_col].iloc[0])
        plateau_thr = best * plateau_frac

        out.append({
            "foil_id": foil,
            "n_runs": int(len(g_sorted)),
            "best_run_score": best,
            f"top{topk}_mean_score": float(top[run_score_col].mean()) if len(top) else np.nan,
            f"top{topk}_std_score": float(top[run_score_col].std(ddof=1)) if len(top) > 1 else 0.0,
            f"top{topk}_iqr_score": float(np.subtract(*np.percentile(top[run_score_col], [75, 25]))) if len(top) else np.nan,
            f"top{topk}_p50_score": float(np.percentile(top[run_score_col], 50)) if len(top) else np.nan,
            f"top{topk}_p90_score": float(np.percentile(top[run_score_col], 90)) if len(top) else np.nan,
            f"top{topk}_min_score": float(top[run_score_col].min()) if len(top) else np.nan,
            f"good_run_pct_>=_{good_thr:.0f}": float((g_sorted[run_score_col] >= good_thr).mean() * 100.0),
            f"bad_run_pct_<=_{bad_thr:.0f}": float((g_sorted[run_score_col] <= bad_thr).mean() * 100.0),
            f"plateau_pct_>=_{plateau_frac:.2f}x_best": float((g_sorted[run_score_col] >= plateau_thr).mean() * 100.0),
        })

    foil_df = pd.DataFrame(out).sort_values(
        by=[f"top{topk}_mean_score", "best_run_score", f"top{topk}_std_score"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return foil_df, good_thr, bad_thr


def pareto_frontier_2d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_higher_better: bool = True,
    y_higher_better: bool = True,
) -> pd.Series:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if not x_higher_better:
        x = -x
    if not y_higher_better:
        y = -y
    n = len(df)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (x[j] >= x[i] and y[j] >= y[i]) and (x[j] > x[i] or y[j] > y[i]):
                nd[i] = False
                break
    return pd.Series(nd, index=df.index)


# =============================
# Data loading
# =============================
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


@st.cache_data(show_spinner=False)
def load_concat_runs(csv_paths: Tuple[str, ...], downsample: int = 5) -> pd.DataFrame:
    """
    Load and concatenate a small set of run CSVs for 'mechanics' diagnostics.
    Adds a 'run_id' column from filename and downsamples rows for plotting.
    """
    frames = []
    for cp in csv_paths:
        try:
            df = load_run_csv(cp)
            df = df.iloc[::max(int(downsample), 1)].copy()
            df["run_id"] = Path(cp).stem
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================
# Feature explanations (editable)
# =============================
FEATURE_DOCS: Dict[str, Dict[str, str]] = {
    "Boat.Speed_kts__mean": {
        "title": "Speed (mean)",
        "what": "Average boat speed over the run (knots).",
        "why": "Primary performance indicator: higher is faster for the same wind/angle/settings.",
        "watch": "Speed alone can hide instability; combine with variability/dynamics.",
    },
    "Boat.Speed_kts__std": {
        "title": "Speed variability (std)",
        "what": "Standard deviation of speed over the run (knots).",
        "why": "Captures short-term oscillations / instability amplitude. Lower is generally better (more stable).",
        "watch": "Very stable but slow can still be undesirable—use with mean speed.",
    },
    "Boat.Speed_kts__max_drawdown_pct": {
        "title": "Speed max drawdown (%)",
        "what": "Largest peak-to-trough speed drop relative to peak during the run.",
        "why": "Tail-risk metric: big collapses are costly in racing (loss of lift, drag spikes, control events). Lower is better.",
        "watch": "Drawdown is sensitive to rare dips; pair with std and stationarity to interpret.",
    },
    "Boat.Speed_kts__acf__integral_timescale_seconds": {
        "title": "Integral timescale (s)",
        "what": "Approximate 'memory' of speed fluctuations computed from the ACF integral.",
        "why": "Separates fast, quickly damped oscillations from slow persistent dynamics. Lower is typically better (faster damping).",
        "watch": "If the run has drift/trend, timescale can inflate; check stationarity.",
    },
    "Boat.Speed_kts__stationarity__mean_drift_ratio": {
        "title": "Stationarity mean drift ratio",
        "what": "Relative change in mean speed across segments of the run.",
        "why": "Detects slow drift (improving or degrading regime). Lower magnitude is better if you want a stable steady-state.",
        "watch": "A small positive drift may reflect convergence; a strong drift indicates non-stationary behavior.",
    },
    "Boat.Speed_kts__stationarity__std_drift_ratio": {
        "title": "Stationarity variance drift ratio",
        "what": "Relative change in speed variability across segments.",
        "why": "Detects if the run becomes more/less unstable with time. Lower is better (stationary variability).",
        "watch": "If variability steadily decreases, that can be good (stabilization). Interpret sign/context.",
    },
    "Boat.Helm__abs_mean": {
        "title": "Helm effort (abs mean)",
        "what": "Average absolute helm command (degrees or simulator units).",
        "why": "Proxy for control effort / corrections needed to hold course. Lower suggests easier handling.",
        "watch": "Low helm could occur with under-responsive dynamics; pair with leeway and speed.",
    },
    "Boat.Leeway__abs_mean": {
        "title": "Leeway (abs mean)",
        "what": "Average absolute leeway angle (deg): sideways drift relative to heading.",
        "why": "Higher leeway implies lateral inefficiency and lost VMG. Lower is typically better.",
        "watch": "Sign conventions vary; abs removes direction so you measure magnitude.",
    },
}

AXIS_DOCS: Dict[str, Dict[str, str]] = {
    "performance": {
        "title": "Axis: Performance",
        "goal": "Go fast for the given wind/angle/settings.",
        "typical_metrics": "Speed mean",
        "interpretation": "Higher axis score = systematically faster runs.",
    },
    "short_term_variability": {
        "title": "Axis: Short-term stability",
        "goal": "Avoid fast oscillations / speed volatility / collapses.",
        "typical_metrics": "Speed std, speed drawdown",
        "interpretation": "Higher axis score = smoother, fewer sudden losses.",
    },
    "slow_dynamics": {
        "title": "Axis: Slow dynamics / damping",
        "goal": "Avoid long-memory regimes and large drift across the run.",
        "typical_metrics": "Integral timescale, mean drift ratio",
        "interpretation": "Higher axis score = quicker damping and more stationary behavior.",
    },
    "control_effort": {
        "title": "Axis: Handling / control",
        "goal": "Minimize helm corrections and lateral losses (leeway).",
        "typical_metrics": "Helm abs mean, leeway abs mean",
        "interpretation": "Higher axis score = easier to steer and more efficient.",
    },
}


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Foil Selection Dashboard", layout="wide")

st.title("Foil Selection Dashboard")
st.caption("Run-level features → Axis scoring → Foil ranking • Interactive decision support")

with st.sidebar:
    st.header("Data")
    features_path = st.text_input("features file (.parquet/.csv)", "data/outputs/features.parquet")

    st.divider()
    st.header("Run → Foil aggregation")
    topk = st.slider("Top-K runs for foil KPIs", 5, 20, 10, 1)
    good_q = st.slider("Good quantile (global)", 0.70, 0.95, 0.85, 0.01)
    bad_q = st.slider("Bad quantile (global)", 0.05, 0.50, 0.30, 0.01)

    st.divider()
    st.header("Axis weights (interactive)")
    st.caption("Weights are normalized to sum to 1.")

    # Default axes + weights
    axes = {
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

    w_perf = st.slider("Performance", 0.0, 1.0, 0.40, 0.01)
    w_var = st.slider("Short-term stability", 0.0, 1.0, 0.25, 0.01)
    w_slow = st.slider("Slow dynamics", 0.0, 1.0, 0.15, 0.01)
    w_ctrl = st.slider("Handling / control", 0.0, 1.0, 0.20, 0.01)

    axis_weights = {
        "performance": w_perf,
        "short_term_variability": w_var,
        "slow_dynamics": w_slow,
        "control_effort": w_ctrl,
    }

    st.divider()
    st.header("Run detail")
    time_col_name = st.text_input("Time column name", "Time")
    downsample = st.slider("Downsample for plots (N)", 1, 20, 2, 1)

# Load features
try:
    features_df = load_df(features_path)
except Exception as e:
    st.error(f"Cannot load features: {e}")
    st.stop()

# Ensure identifiers exist (best effort)
for c in ["foil_id", "script_id"]:
    if c not in features_df.columns:
        st.error(f"Missing required column in features_df: {c}")
        st.stop()

# Build scored_df (interactive)
scored_df = build_scored_df(features_df, axes=axes, axis_weights=axis_weights)
foil_rank_df, good_thr, bad_thr = rank_foils_from_runs(
    scored_df, topk=topk, good_quantile=good_q, bad_quantile=bad_q
)

# Tabs
tab_overview, tab_features, tab_foil, tab_run, tab_decision = st.tabs(
    ["Overview", "Features & Methodology", "Foil detail", "Run detail", "Decision"]
)


# =============================
# Overview
# =============================
with tab_overview:
    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="top")

    with c1:
        st.subheader("Foil ranking (aggregated)")
        st.caption(f"Good threshold = {good_thr:.2f} (q={good_q:.2f}) • Bad threshold = {bad_thr:.2f} (q={bad_q:.2f})")
        st.dataframe(foil_rank_df, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Pareto: Performance vs Robustness")
        st.caption("x = topK mean score (higher better) • y = topK std (lower better)")

        mean_col = f"top{topk}_mean_score"
        std_col = f"top{topk}_std_score"
        tmp = foil_rank_df.copy()
        tmp["pareto"] = pareto_frontier_2d(tmp, mean_col, std_col, x_higher_better=True, y_higher_better=False)

        chart = (
            alt.Chart(tmp)
            .mark_circle(size=180)
            .encode(
                x=alt.X(f"{mean_col}:Q", title=f"Top{topk} mean run_score"),
                y=alt.Y(f"{std_col}:Q", title=f"Top{topk} std (lower = robust)"),
                color=alt.Color("pareto:N", title="Pareto", scale=alt.Scale(domain=[False, True])),
                tooltip=["foil_id", "n_runs", "best_run_score", mean_col, std_col],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

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

    st.divider()
    st.subheader("Weight sensitivity (quick view)")
    st.write(
        "Use the sliders in the sidebar to change axis weights. "
        "Rankings and Pareto plot update immediately. This is a simple decision-sensitivity analysis."
    )


# =============================
# Features & Methodology
# =============================
with tab_features:
    st.subheader("How the platform works")
    st.markdown(
        """
**Pipeline**
1) We compute run-level features from each simulation (one CSV = one run).
2) We normalize each feature across all runs using percentile ranks (0..1).
   - For “higher is better” features: higher percentile → higher score.
   - For “lower is better” features (std, drawdown, leeway, etc.): we invert (1 - percentile).
3) We group features into a small set of conceptual axes (performance / stability / dynamics / handling).
   Each axis score is the mean of its feature scores (per run).
4) We compute a weighted sum of axis scores to obtain `run_score` (0..100).
5) We aggregate run_scores per foil (topK mean, robustness, tails) to support selection.
"""
    )

    st.markdown("### Axis definitions")
    for axis_name, doc in AXIS_DOCS.items():
        with st.expander(doc["title"], expanded=False):
            st.write(f"**Goal:** {doc['goal']}")
            st.write(f"**Typical metrics:** {doc['typical_metrics']}")
            st.write(f"**Interpretation:** {doc['interpretation']}")
            # Show which metrics are currently used in this axis (from `axes`)
            current = axes.get(axis_name, [])
            used_cols = [m["col"] for m in current if m["col"] in scored_df.columns]
            if used_cols:
                st.write("**Metrics used in this app:**")
                for c in used_cols:
                    st.code(c, language="text")
            else:
                st.warning("No metric columns found for this axis in your features file.")

    st.divider()
    st.markdown("### Feature dictionary (run-level)")
    st.write("Each feature below is computed per run and then normalized across all runs.")

    # Show only the features that exist in the current dataframe
    existing_features = [k for k in FEATURE_DOCS.keys() if k in scored_df.columns]
    if not existing_features:
        st.warning("No documented feature columns found in scored_df. Update FEATURE_DOCS to match your column names.")
    else:
        for feat in existing_features:
            d = FEATURE_DOCS[feat]
            with st.expander(f"{d['title']} — `{feat}`", expanded=False):
                st.write(f"**What it is:** {d['what']}")
                st.write(f"**Why it matters:** {d['why']}")
                st.write(f"**Watch-outs:** {d['watch']}")

    st.divider()
    st.markdown("### Practical interpretation: what to look for")
    st.markdown(
        """
- **High performance + high stability** → attractive racing settings.
- **High performance but poor handling** → potentially “pointy” foil / more skipper workload.
- **Strong slow-dynamics penalties** → long-memory / drifting regimes; risky in real conditions.
- **Robust foils** are those that maintain high scores for many settings (wide good tail, small bad tail).
"""
    )


# =============================
# Foil detail
# =============================
with tab_foil:
    st.subheader("Foil deep dive")
    foil_ids = sorted(scored_df["foil_id"].dropna().astype(str).unique())
    sel_foil = st.selectbox("Select foil", foil_ids, index=0)

    g = scored_df[scored_df["foil_id"].astype(str) == str(sel_foil)].dropna(subset=["run_score"]).copy()
    g = g.sort_values("run_score", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Runs", int(len(g)))
    with k2:
        st.metric("Best run_score", f"{g['run_score'].max():.2f}")
    with k3:
        st.metric(f"Top{topk} mean", f"{g.head(topk)['run_score'].mean():.2f}")
    with k4:
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

        st.markdown("### Top runs")
        show_cols = [c for c in ["script_id", "run_score", "csv_path"] if c in g.columns]
        st.dataframe(g[show_cols].head(20), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Axis profile (median over runs)")
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
            st.info("No axis score columns found.")

        st.markdown("### Tails (global thresholds)")
        good_pct = float((g["run_score"] >= good_thr).mean() * 100)
        bad_pct = float((g["run_score"] <= bad_thr).mean() * 100)
        st.write(f"- Good runs (>= {good_thr:.2f}): **{good_pct:.1f}%**")
        st.write(f"- Bad runs (<= {bad_thr:.2f}): **{bad_pct:.1f}%**")

    st.divider()
    st.markdown("### Parameter grid (if present in features)")
    param_candidates = ["Boat.Aero.Travel", "Boat.Keel.KeelCant", "Boat.Port.FoilRake", "Boat.Stbd.FoilRake", "Boat.TWS_kts"]
    existing_params = [p for p in param_candidates if p in g.columns]

    if existing_params:
        pivot_cols = existing_params[:3]
        grp = g.groupby(pivot_cols, dropna=False)["run_score"].agg(["count", "mean", "max"]).reset_index()
        grp = grp.sort_values("max", ascending=False)
        st.dataframe(grp.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("No parameter columns found in your features file. If you want this, include them in features_df.")



    st.divider()
    st.markdown("### Mechanics diagnostics (raw time-series, optional)")
    st.caption(
        "These plots use the *raw* run CSVs (a few selected runs) to visualize the relationships you discussed: "
        "Speed↔Leeway, Heel distribution, Trim distribution, and VMG↔Speed. "
        "This helps distinguish *potential* vs *robustness* and diagnose why a foil behaves the way it does."
    )

    if "csv_path" not in g.columns or g["csv_path"].isna().all():
        st.info("No `csv_path` column available for this foil, so raw time-series diagnostics cannot be loaded.")
    else:
        d1, d2, d3 = st.columns([0.40, 0.30, 0.30])
        with d1:
            n_runs_load = st.slider("Number of runs to load (for diagnostics)", 1, 8, 4, 1)
        with d2:
            pick_mode = st.selectbox("Which runs?", ["Top score", "Random sample", "Worst score"], index=0)
        with d3:
            diag_downsample = st.slider("Downsample (raw plots)", 1, 20, max(downsample, 5), 1)

        # Choose runs
        paths = g.dropna(subset=["csv_path"])[["csv_path", "run_score"]].copy()
        if pick_mode == "Top score":
            chosen = paths.sort_values("run_score", ascending=False).head(n_runs_load)
        elif pick_mode == "Worst score":
            chosen = paths.sort_values("run_score", ascending=True).head(n_runs_load)
        else:
            chosen = paths.sample(min(n_runs_load, len(paths)), random_state=42) if len(paths) else paths.head(0)

        csv_paths = tuple(chosen["csv_path"].astype(str).tolist())
        raw = load_concat_runs(csv_paths, downsample=diag_downsample)

        if raw.empty:
            st.warning("Could not load any run CSVs for diagnostics (paths missing or unreadable).")
        else:
            # Coerce numeric
            for c in ["Boat.Speed_kts", "Boat.Leeway", "Boat.Heel", "Boat.Trim", "Boat.VMG_kts"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")

            # 1) Speed vs Leeway (scatter)
            if "Boat.Speed_kts" in raw.columns and "Boat.Leeway" in raw.columns:
                st.markdown("#### Speed vs Leeway (raw points)")
                tmp = raw.dropna(subset=["Boat.Speed_kts", "Boat.Leeway"]).copy()
                # If leeway is signed, show magnitude option
                show_abs = st.checkbox("Show |Leeway| (magnitude)", value=True, key=f"abs_leeway_{sel_foil}")
                if show_abs:
                    tmp["Leeway_plot"] = tmp["Boat.Leeway"].abs()
                    leeway_title = "|Leeway|"
                else:
                    tmp["Leeway_plot"] = tmp["Boat.Leeway"]
                    leeway_title = "Leeway"
                sc = (
                    alt.Chart(tmp)
                    .mark_circle(opacity=0.25, size=18)
                    .encode(
                        x=alt.X("Boat.Speed_kts:Q", title="Boat.Speed_kts"),
                        y=alt.Y("Leeway_plot:Q", title=leeway_title),
                        color=alt.Color("run_id:N", title="run"),
                        tooltip=["run_id", alt.Tooltip("Boat.Speed_kts:Q", format=".2f"), alt.Tooltip("Leeway_plot:Q", format=".3f")],
                    )
                    .interactive()
                )
                st.altair_chart(sc, use_container_width=True)

            # 2) Heel distribution
            if "Boat.Heel" in raw.columns:
                st.markdown("#### Heel distribution (raw)")
                tmp = raw.dropna(subset=["Boat.Heel"])
                hist = (
                    alt.Chart(tmp)
                    .mark_bar()
                    .encode(
                        x=alt.X("Boat.Heel:Q", bin=alt.Bin(maxbins=50), title="Boat.Heel"),
                        y=alt.Y("count():Q", title="count"),
                        tooltip=[alt.Tooltip("count():Q")],
                    )
                )
                st.altair_chart(hist, use_container_width=True)

            # 3) Trim distribution
            if "Boat.Trim" in raw.columns:
                st.markdown("#### Trim distribution (raw)")
                tmp = raw.dropna(subset=["Boat.Trim"])
                hist = (
                    alt.Chart(tmp)
                    .mark_bar()
                    .encode(
                        x=alt.X("Boat.Trim:Q", bin=alt.Bin(maxbins=50), title="Boat.Trim"),
                        y=alt.Y("count():Q", title="count"),
                        tooltip=[alt.Tooltip("count():Q")],
                    )
                )
                st.altair_chart(hist, use_container_width=True)

            # 4) VMG vs Speed
            if "Boat.Speed_kts" in raw.columns and "Boat.VMG_kts" in raw.columns:
                st.markdown("#### VMG_kts vs Speed (raw points)")
                tmp = raw.dropna(subset=["Boat.Speed_kts", "Boat.VMG_kts"]).copy()
                sc = (
                    alt.Chart(tmp)
                    .mark_circle(opacity=0.25, size=18)
                    .encode(
                        x=alt.X("Boat.Speed_kts:Q", title="Boat.Speed_kts"),
                        y=alt.Y("Boat.VMG_kts:Q", title="Boat.VMG_kts"),
                        color=alt.Color("run_id:N", title="run"),
                        tooltip=["run_id", alt.Tooltip("Boat.Speed_kts:Q", format=".2f"), alt.Tooltip("Boat.VMG_kts:Q", format=".2f")],
                    )
                    .interactive()
                )
                st.altair_chart(sc, use_container_width=True)

            # Quick per-run stability summary from raw series (optional)
            st.markdown("#### Quick stability summary (from loaded raw runs)")
            summary_rows = []
            for rid, rr in raw.groupby("run_id"):
                row = {"run_id": rid, "n_points": int(len(rr))}
                for col in ["Boat.Speed_kts", "Boat.Leeway", "Boat.Heel", "Boat.Trim", "Boat.VMG_kts"]:
                    if col in rr.columns:
                        v = pd.to_numeric(rr[col], errors="coerce")
                        row[f"{col}__mean"] = float(v.mean()) if np.isfinite(v.mean()) else np.nan
                        row[f"{col}__std"] = float(v.std(ddof=1)) if len(v.dropna()) > 1 else 0.0
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows).sort_values("Boat.Speed_kts__mean", ascending=False), use_container_width=True, hide_index=True)


# =============================
# Run detail
# =============================
with tab_run:
    st.subheader("Run deep dive (time series)")

    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        foil_for_run = st.selectbox("Foil", sorted(scored_df["foil_id"].dropna().astype(str).unique()), index=0, key="run_foil")
    g2 = scored_df[scored_df["foil_id"].astype(str) == str(foil_for_run)].dropna(subset=["run_score"]).copy()
    g2 = g2.sort_values("run_score", ascending=False)

    with c2:
        run_choices = g2["script_id"].astype(str).tolist() if "script_id" in g2.columns else []
        if not run_choices:
            st.warning("No script_id column found; cannot select runs.")
            st.stop()
        sel_script = st.selectbox("Run (script_id)", run_choices, index=0, key="run_id")

    run_row = g2[g2["script_id"].astype(str) == str(sel_script)].head(1)
    if run_row.empty:
        st.warning("Run not found.")
        st.stop()

    st.metric("run_score", f"{float(run_row['run_score'].iloc[0]):.2f}")

    # Axis breakdown
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

    if "csv_path" not in run_row.columns or pd.isna(run_row["csv_path"].iloc[0]):
        st.error("No csv_path found for this run in features/scored data.")
        st.stop()

    csv_path = str(run_row["csv_path"].iloc[0])
    st.write(f"CSV: `{csv_path}`")

    try:
        run_df = load_run_csv(csv_path)
    except Exception as e:
        st.error(f"Cannot load run CSV: {e}")
        st.stop()

    # Time column
    if time_col_name not in run_df.columns:
        run_df[time_col_name] = np.arange(len(run_df), dtype=float)

    # Select series to plot
    default_series = [c for c in ["Boat.Speed_kts", "Boat.Helm", "Boat.Trim", "Boat.Leeway", "Boat.TWA"] if c in run_df.columns]
    series = st.multiselect("Series", options=list(run_df.columns), default=default_series)

    plot_df = run_df.iloc[::downsample].copy()

    if series:
        long = plot_df[[time_col_name] + series].copy()
        for c in series:
            long[c] = pd.to_numeric(long[c], errors="coerce")
        long = long.melt(id_vars=[time_col_name], var_name="series", value_name="value").dropna(subset=["value"])

        ts_chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X(f"{time_col_name}:Q", title="Time"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("series:N"),
                tooltip=[time_col_name, "series", "value"],
            )
            .interactive()
        )
        st.altair_chart(ts_chart, use_container_width=True)
    else:
        st.info("Select at least one series to plot.")

    st.divider()
    st.markdown("### Run KPIs (from features/scoring)")
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
        st.info("No KPI columns found (update kpi_cols to match your feature names).")


# =============================
# Decision tab
# =============================
with tab_decision:
    st.subheader("Decision support")
    st.write(
        "This page summarizes *what we would recommend* given the current axis weights and the observed trade-offs."
    )

    mean_col = f"top{topk}_mean_score"
    std_col = f"top{topk}_std_score"
    good_col = f"good_run_pct_>=_{good_thr:.0f}"
    bad_col = f"bad_run_pct_<=_{bad_thr:.0f}"

    tmp = foil_rank_df.copy()
    tmp["pareto_perf_robust"] = pareto_frontier_2d(tmp, mean_col, std_col, True, False)
    tmp["pareto_good_bad"] = pareto_frontier_2d(tmp, good_col, bad_col, True, False) if good_col in tmp.columns and bad_col in tmp.columns else False

    st.markdown("### Candidates on Pareto frontiers")
    st.caption("We compute two Pareto views: (1) performance vs robustness and (2) good-tail vs bad-tail (risk).")
    show = ["foil_id", "n_runs", "best_run_score", mean_col, std_col]
    if good_col in tmp.columns: show.append(good_col)
    if bad_col in tmp.columns: show.append(bad_col)
    show += ["pareto_perf_robust", "pareto_good_bad"]
    st.dataframe(tmp[show], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Recommendation (simple and defensible)")
    st.write(
        "A practical recommendation rule is:\n"
        "- Start from the Pareto set (not dominated)\n"
        "- Prefer higher topK mean score\n"
        "- Break ties using lower topK std (robustness)\n"
        "- Finally use tails: higher good% and lower bad%"
    )

    # Build a simple decision score (foil-level) derived from aggregated KPIs
    # NOTE: This is *not* the run_score; it's a decision summary for the presentation.
    decision_df = tmp.copy()

    # Normalize foil KPIs to 0..1 for display (percentile across foils)
    def foil_pct(col, higher_better=True):
        s = pd.to_numeric(decision_df[col], errors="coerce")
        pct = s.rank(pct=True, method="average")
        return pct if higher_better else (1 - pct)

    decision_df["dec__perf"] = foil_pct(mean_col, True)
    decision_df["dec__robust"] = foil_pct(std_col, False)
    if good_col in decision_df.columns:
        decision_df["dec__good"] = foil_pct(good_col, True)
    else:
        decision_df["dec__good"] = np.nan
    if bad_col in decision_df.columns:
        decision_df["dec__bad"] = foil_pct(bad_col, False)  # lower bad% is better
    else:
        decision_df["dec__bad"] = np.nan

    # Simple decision score: emphasize performance & robustness, tails as secondary
    decision_df["decision_score"] = (
        0.45 * decision_df["dec__perf"]
        + 0.35 * decision_df["dec__robust"]
        + 0.10 * decision_df["dec__good"].fillna(0)
        + 0.10 * decision_df["dec__bad"].fillna(0)
    ) * 100

    decision_df = decision_df.sort_values("decision_score", ascending=False)

    st.markdown("### Decision scoreboard (foil-level)")
    st.dataframe(
        decision_df[["foil_id", "decision_score", mean_col, std_col] + ([good_col] if good_col in decision_df.columns else []) + ([bad_col] if bad_col in decision_df.columns else []) + ["pareto_perf_robust"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### Why the ranking changes when you move weights")
    st.write(
        "Your axis weights represent *preferences*:\n"
        "- Increasing **Performance** favors foils with higher average speed.\n"
        "- Increasing **Short-term stability** favors foils with smoother speed (low std / drawdown).\n"
        "- Increasing **Slow dynamics** penalizes long-memory or drifting regimes.\n"
        "- Increasing **Handling** favors foils needing less helm and exhibiting less leeway.\n\n"
        "Use the sidebar sliders to explore these trade-offs and communicate them clearly during the interview."
    )


# Footer
st.caption(
    "Tip: For the interview, keep the story simple: (1) define axes, (2) show Pareto trade-offs, "
    "(3) explain robustness with tails, (4) show sensitivity to weights."
)