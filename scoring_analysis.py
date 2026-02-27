import numpy as np
import pandas as pd


# -----------------------------
# Percentile scoring
# -----------------------------
def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Convert a numeric series to percentile scores in [0,1].
    - higher_is_better=True  -> higher values get higher scores
    - higher_is_better=False -> lower values get higher scores

    Uses rank(pct=True) which is robust-ish and simple.
    """
    s = pd.to_numeric(series, errors="coerce")
    # pct rank among non-NaNs
    pct = s.rank(pct=True, method="average")
    if higher_is_better:
        return pct
    else:
        return 1.0 - pct
    

def build_run_scores(
    features_df: pd.DataFrame,
    id_cols=("foil_id", "script_id"),
    axes=None,
    axis_weights=None,
    score_scale: float = 100.0,
) -> pd.DataFrame:
    """
    Build run scores using *axes* (groups of metrics).
    - Each metric is percentile-normalized to [0,1] with correct direction.
    - Each axis gets a weight (importance).
    - Within an axis, the axis weight is split equally among available metrics per row.

    Parameters
    ----------
    features_df : DataFrame
        One row per run. Must contain id_cols + metric columns.
    axes : dict
        {
          "axis_name": [
             {"col": "...", "higher_is_better": True/False},
             ...
          ],
          ...
        }
    axis_weights : dict
        {"axis_name": weight, ...}  (weights don't need to sum to 1; we normalize)
    """
    df = features_df.copy()

    # Default axes (conceptually orthogonal)
    # - Performance level
    # - Short-term variability
    # - Slow dynamics / persistence
    # - Control / lateral inefficiency
    #
    # You can edit this list based on columns present.
    if axes is None:
        axes = {
            "performance": [
                {"col": "Boat.Speed_kts__mean", "higher_is_better": True},
            ],
            "short_term_variability": [
                {"col": "Boat.Speed_kts__std", "higher_is_better": False},
                # Use either drawdown OR std if you want less redundancy.
                {"col": "Boat.Speed_kts__max_drawdown_pct", "higher_is_better": False},
            ],
            "slow_dynamics": [
                {"col": "Boat.Speed_kts__acf__integral_timescale_seconds", "higher_is_better": False},
                # Pick ONE of the two drift ratios if you want: both can be redundant.
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

        axis_weights = {
            "performance": 0.25,
            "short_term_variability": 0.25,
            "slow_dynamics": 0.25,
            "control_effort": 0.25,
        }

    # Normalize axis weights to sum to 1 (optional but convenient)
    aw = pd.Series(axis_weights, dtype=float)
    aw = aw[aw > 0]
    if aw.sum() <= 0:
        raise ValueError("axis_weights must have positive weights.")
    aw = aw / aw.sum()

    # Create metric score columns
    used_metric_cols = []
    for axis_name, metrics in axes.items():
        for spec in metrics:
            col = spec["col"]
            if col not in df.columns:
                continue
            score_col = f"{col}__score"
            if score_col not in df.columns:
                df[score_col] = percentile_score(df[col], spec["higher_is_better"])
            used_metric_cols.append(col)

    # Compute axis scores: average of metric scores available in that axis per row
    axis_score_cols = []
    for axis_name, metrics in axes.items():
        # only keep metrics that exist
        cols = [spec["col"] for spec in metrics if spec["col"] in df.columns]
        score_cols = [f"{c}__score" for c in cols]
        if len(score_cols) == 0:
            df[f"axis__{axis_name}__score"] = np.nan
            axis_score_cols.append(f"axis__{axis_name}__score")
            continue

        S = df[score_cols].to_numpy(dtype=float)
        valid = np.isfinite(S)
        # row-wise mean of valid metrics
        denom = valid.sum(axis=1)
        axis_score = np.where(denom > 0, np.nansum(S, axis=1) / denom, np.nan)

        df[f"axis__{axis_name}__score"] = axis_score
        axis_score_cols.append(f"axis__{axis_name}__score")

    # Global run_score = sum(axis_weight * axis_score) with per-row renorm if some axes missing
    A = df[[f"axis__{k}__score" for k in aw.index]].to_numpy(dtype=float)
    W = aw.values
    W_mat = np.tile(W, (len(df), 1))

    validA = np.isfinite(A)
    W_eff = np.where(validA, W_mat, 0.0)
    denom = W_eff.sum(axis=1)
    num = np.nansum(A * W_eff, axis=1)
    run_score_01 = np.where(denom > 0, num / denom, np.nan)
    df["run_score"] = run_score_01 * score_scale

    # Return a compact df
    keep = list(id_cols)

    # add raw metrics + metric scores used
    keep_metrics = sorted(set(used_metric_cols))
    keep += [c for c in keep_metrics if c in df.columns]
    keep += [f"{c}__score" for c in keep_metrics if f"{c}__score" in df.columns]

    # add axis scores + run score
    keep += [f"axis__{k}__score" for k in aw.index]
    keep += ["run_score"]

    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def build_run_scores2(
    features_df: pd.DataFrame,
    id_cols=("foil_id", "script_id"),
    metric_specs=None,
    weights=None,
    score_scale: float = 100.0,
) -> pd.DataFrame:
    """
    Build normalized criterion scores + a global score for each run.

    Parameters
    ----------
    features_df : DataFrame
        Must include id_cols + metric columns.
    id_cols : tuple
        Columns identifying the run (foil_id, script_id).
    metric_specs : dict
        {metric_col: {"higher_is_better": bool}}
    weights : dict
        {metric_col: weight} weights sum doesn't need to be 1 (we normalize).
    score_scale : float
        Global score range, default 0..100.

    Returns
    -------
    scored_df : DataFrame
        Contains id_cols, raw metrics, normalized metric scores (suffix "__score"),
        and "run_score" (0..score_scale).
    """
    df = features_df.copy()

    if metric_specs is None:
        # Default spec using the columns you have been producing
        metric_specs = {
            "Boat.Speed_kts__mean": {"higher_is_better": True},
            "Boat.Speed_kts__std": {"higher_is_better": False},
            "Boat.Speed_kts__max_drawdown_pct": {"higher_is_better": False},
            "Boat.Helm__abs_mean": {"higher_is_better": False},
            "Boat.Leeway__abs_mean": {"higher_is_better": False},
            "Boat.Speed_kts__acf__integral_timescale_seconds": {"higher_is_better": False},
            "Boat.Speed_kts__stationarity__mean_drift_ratio": {"higher_is_better": False},
            "Boat.Speed_kts__stationarity__std_drift_ratio": {"higher_is_better": False},
        }

    # If stationarity ratios are currently only in a nested dict, you can pre-flatten them
    # But here we assume they are already columns. If not, see helper below.

    if weights is None:
        # A defensible “balanced” weighting
        weights = {
            "Boat.Speed_kts__mean": 0.40,
            "Boat.Speed_kts__std": 0.15,
            "Boat.Speed_kts__max_drawdown_pct": 0.15,
            "Boat.Helm__abs_mean": 0.10,
            "Boat.Leeway__abs_mean": 0.10,
            "Boat.Speed_kts__acf__integral_timescale_seconds": 0.05,
            "Boat.Speed_kts__stationarity__mean_drift_ratio": 0.03,
            "Boat.Speed_kts__stationarity__std_drift_ratio": 0.02,
        }

        weights = {
            "Boat.Speed_kts__mean": 0.125,
            "Boat.Speed_kts__std": 0.125,
            "Boat.Speed_kts__max_drawdown_pct": 0.125,
            "Boat.Helm__abs_mean": 0.125,
            "Boat.Leeway__abs_mean": 0.125,
            "Boat.Speed_kts__acf__integral_timescale_seconds": 0.125,
            "Boat.Speed_kts__stationarity__mean_drift_ratio": 0.125,
            "Boat.Speed_kts__stationarity__std_drift_ratio": 0.125,
        }

    # Keep only metrics that exist
    metric_cols = [m for m in metric_specs.keys() if m in df.columns]
    if len(metric_cols) == 0:
        raise ValueError("None of the metric columns in metric_specs exist in features_df.")

    # Create normalized scores per metric
    for m in metric_cols:
        hib = metric_specs[m]["higher_is_better"]
        df[m + "__score"] = percentile_score(df[m], higher_is_better=hib)

    # Weighted aggregation with NaN-safe behavior:
    # - if a run is missing a metric -> ignore that metric weight for that run
    w = pd.Series({m: weights.get(m, 0.0) for m in metric_cols}, dtype=float)
    w = w[w > 0]
    used_metrics = list(w.index)

    score_cols = [m + "__score" for m in used_metrics]
    W = w.values

    # Compute per-row weighted mean with dynamic renormalization if NaNs present
    S = df[score_cols].to_numpy(dtype=float)
    W_mat = np.tile(W, (len(df), 1))

    valid = np.isfinite(S)
    W_eff = np.where(valid, W_mat, 0.0)
    denom = W_eff.sum(axis=1)

    # avoid division by 0
    num = np.nansum(S * W_eff, axis=1)
    run_score_01 = np.where(denom > 0, num / denom, np.nan)

    df["run_score"] = run_score_01 * score_scale

    # Return a compact view: id + raw + scores + global
    keep = list(id_cols) + metric_cols + [m + "__score" for m in metric_cols] + ["run_score"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# -----------------------------
# Foil-level aggregation
# -----------------------------
def rank_foils_from_runs(
    scored_df: pd.DataFrame,
    foil_col: str = "foil_id",
    run_score_col: str = "run_score",
    topk: int = 10,
    good_threshold: float = 90.0,
) -> pd.DataFrame:
    """
    Aggregate run scores into foil-level KPIs.

    Returns foil_rank_df with:
      - n_runs
      - best_run_score
      - topk_mean_score
      - topk_std_score (robustness; lower is better)
      - good_run_pct (percent of runs >= threshold)
      - topk_min_score (how bad is the worst among the topk)
    """
    if foil_col not in scored_df.columns:
        raise KeyError(f"'{foil_col}' not in scored_df")
    if run_score_col not in scored_df.columns:
        raise KeyError(f"'{run_score_col}' not in scored_df")

    df = scored_df.dropna(subset=[run_score_col]).copy()

    good_threshold = df["run_score"].quantile(0.85)
    bad_threshold = scored_df["run_score"].quantile(0.30)

    out_rows = []
    for foil, g in df.groupby(foil_col):
        g_sorted = g.sort_values(run_score_col, ascending=False)
        top = g_sorted.head(topk)

        n = len(g_sorted)
        best = float(g_sorted[run_score_col].iloc[0])
        top_mean = float(top[run_score_col].mean()) if len(top) else np.nan
        top_std = float(top[run_score_col].std(ddof=1)) if len(top) > 1 else 0.0
        top_min = float(top[run_score_col].min()) if len(top) else np.nan
        good_pct = float((g_sorted[run_score_col] >= good_threshold).mean()) * 100.0
        bad_pct = float((g_sorted[run_score_col] <= bad_threshold).mean()) * 100.0

        out_rows.append({
            foil_col: foil,
            "n_runs": n,
            "best_run_score": best,
            f"top{topk}_mean_score": top_mean,
            f"top{topk}_std_score": top_std,
            f"top{topk}_min_score": top_min,
            f"good_run_pct_>=_{good_threshold:.0f}": good_pct,
            f"bad_run_pct_>=_{bad_threshold:.0f}": bad_pct,
        })

    foil_rank_df = pd.DataFrame(out_rows)

    # A nice default ranking: prioritize topK mean (exploitable perf), then best run, then robustness
    foil_rank_df = foil_rank_df.sort_values(
        by=[f"top{topk}_mean_score", "best_run_score", f"top{topk}_std_score"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return foil_rank_df


# -----------------------------
# Optional: helper to flatten stationarity dict into columns
# -----------------------------
def flatten_stationarity_into_features(
    features_df: pd.DataFrame,
    stationarity_dict_col: str = "stationarity",
    prefix_map=None,
) -> pd.DataFrame:
    """
    If you stored stationarity reports as nested dicts per run (e.g. result["stationarity"]),
    this helper flattens Speed stationarity into columns like:
      Boat.Speed_kts__stationarity__mean_drift_ratio
      Boat.Speed_kts__stationarity__std_drift_ratio

    Expects each cell in stationarity_dict_col to be a dict containing keys for series,
    e.g. stationarity["Boat.Speed_kts"]["mean_drift_ratio"].
    """
    df = features_df.copy()
    if stationarity_dict_col not in df.columns:
        return df

    if prefix_map is None:
        prefix_map = {
            "Boat.Speed_kts": "Boat.Speed_kts__stationarity__",
            "Boat.Helm": "Boat.Helm__stationarity__",
            "Boat.Trim": "Boat.Trim__stationarity__",
            "Boat.Leeway": "Boat.Leeway__stationarity__",
        }

    def extract(row_dict, series_name, key):
        try:
            return row_dict.get(series_name, {}).get(key, np.nan)
        except Exception:
            return np.nan

    for series_name, pref in prefix_map.items():
        df[pref + "mean_drift_ratio"] = df[stationarity_dict_col].apply(lambda d: extract(d, series_name, "mean_drift_ratio"))
        df[pref + "std_drift_ratio"] = df[stationarity_dict_col].apply(lambda d: extract(d, series_name, "std_drift_ratio"))
        df[pref + "slope_per_sample"] = df[stationarity_dict_col].apply(lambda d: extract(d, series_name, "slope_per_sample"))

    return df


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Suppose you already built a features_df with one row per run, including foil_id and script_id.
    features_df = pd.read_parquet("data/outputs/features.parquet")  # or csv
    # If stationarity is nested dict in a column:
    # features_df = flatten_stationarity_into_features(features_df, stationarity_dict_col="stationarity")

    scored_df = build_run_scores(features_df)
    foil_rank_df = rank_foils_from_runs(scored_df, topk=10, good_threshold=90.0)

    print(scored_df.sort_values("run_score", ascending=False).head(15))
    print(foil_rank_df.head(10))
    pass