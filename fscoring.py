# scoring.py
# Scoring with your 4-block structure + my recommendations (levels separated).
#
# run_score  : blocks 1-3 (run-level only)
# foil_score : combine mean_run_score + block4 robustness (foil-level)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True, method="average")
    return pct if higher_is_better else (1.0 - pct)


@dataclass(frozen=True)
class MetricSpec:
    col: str
    higher_is_better: bool


# --- Your blocks (with recommendations) ---
RUN_BLOCKS: Dict[str, List[MetricSpec]] = {
    "performance_useful": [
        MetricSpec("VMG_mean", True),
        MetricSpec("VMG_95p", True),
    ],
    "stability_intrarun": [
        MetricSpec("IQR_Speed", False),
        MetricSpec("IQR_Trim", False),
        MetricSpec("IQR_Heel", False),
    ],
    "dynamic_degradation": [
        # Recommendation: delta computed on % windows in feature engineering
        MetricSpec("Delta_Speed_LastFirst", True),
        MetricSpec("MaxDrawdown_Speed_pct", False),
        MetricSpec("Slope_AbsLeeway", False),
    ],
}

FOIL_BLOCKS: Dict[str, List[MetricSpec]] = {
    "robustness_interrun": [
        MetricSpec("Plateau_pct", True),
        # Recommendation: prefer robust dispersion; default to IQR
        MetricSpec("InterRun_VMG_IQR", False),
        # If you prefer MAD, swap to InterRun_VMG_MAD
    ]
}


def add_metric_scores(df: pd.DataFrame, blocks: Dict[str, List[MetricSpec]]) -> pd.DataFrame:
    out = df.copy()
    for _, specs in blocks.items():
        for sp in specs:
            if sp.col not in out.columns:
                continue
            sc = f"{sp.col}__score"
            if sc not in out.columns:
                out[sc] = percentile_score(out[sp.col], sp.higher_is_better)
    return out


def score_by_blocks(
    df: pd.DataFrame,
    blocks: Dict[str, List[MetricSpec]],
    block_weights: Dict[str, float],
    prefix: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute a weighted score from blocks.
    Each block weight is divided equally among available metrics *per row*.
    Returns df with block scores + global score column f"{prefix}_score".
    """
    out = df.copy()

    # normalize weights (ignore missing/zero)
    bw = pd.Series({k: float(v) for k, v in block_weights.items() if float(v) > 0}, dtype=float)
    if bw.sum() <= 0:
        out[f"{prefix}_score"] = np.nan
        return out, []
    bw = bw / bw.sum()

    used_blocks = [b for b in bw.index if b in blocks]

    block_score_cols = []
    for b in used_blocks:
        specs = blocks[b]
        score_cols = [f"{sp.col}__score" for sp in specs if f"{sp.col}__score" in out.columns]
        colname = f"{prefix}__block__{b}__score"
        block_score_cols.append(colname)

        if not score_cols:
            out[colname] = np.nan
            continue

        S = out[score_cols].to_numpy(dtype=float)
        valid = np.isfinite(S)
        denom = valid.sum(axis=1)
        out[colname] = np.where(denom > 0, np.nansum(S, axis=1) / denom, np.nan)

    # weighted aggregation with per-row renormalization (if some block scores missing)
    B = out[block_score_cols].to_numpy(dtype=float)
    W = bw.loc[used_blocks].to_numpy(dtype=float)
    W_mat = np.tile(W, (len(out), 1))

    validB = np.isfinite(B)
    W_eff = np.where(validB, W_mat, 0.0)
    denom = W_eff.sum(axis=1)
    num = np.nansum(B * W_eff, axis=1)

    out[f"{prefix}_score"] = np.where(denom > 0, num / denom, np.nan) * 100.0
    return out, block_score_cols


def build_run_and_foil_scores(
    features_df: pd.DataFrame,
    run_block_weights: Dict[str, float],
    foil_mix_weights: Dict[str, float],
    foil_block_weights: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - scored_runs: one row per run with run_score (+ block breakdown)
      - scored_foils: one row per foil with foil_score (+ breakdown)
    """
    if foil_block_weights is None:
        foil_block_weights = {"robustness_interrun": 1.0}  # single block anyway

    # --- Run-level scoring (Blocks 1-3) ---
    df = add_metric_scores(features_df, RUN_BLOCKS)
    df, run_block_cols = score_by_blocks(df, RUN_BLOCKS, run_block_weights, prefix="run")

    # --- Foil-level aggregation ---
    # Aggregate run_score per foil (exploitable performance over all settings)
    agg = df.groupby("foil_id", dropna=False).agg(
        n_runs=("script_id", "count"),
        mean_run_score=("run_score", "mean"),
        top10_mean_run_score=("run_score", lambda s: float(np.mean(np.sort(s.dropna())[-10:])) if s.dropna().size else np.nan),
        top10_std_run_score=("run_score", lambda s: float(np.std(np.sort(s.dropna())[-10:], ddof=1)) if s.dropna().size >= 2 else 0.0),
    ).reset_index()

    # Take foil-level block columns from first row (they are constant per foil in features_df)
    foil_cols = ["Plateau_pct", "InterRun_VMG_IQR", "InterRun_VMG_MAD", "VMG_ref_95"]
    firsts = df.groupby("foil_id", dropna=False)[foil_cols].first().reset_index()
    foils = agg.merge(firsts, on="foil_id", how="left")

    # Robustness block scoring (Block 4)
    foils = add_metric_scores(foils, FOIL_BLOCKS)
    foils, foil_block_cols = score_by_blocks(foils, FOIL_BLOCKS, foil_block_weights, prefix="foilblock")

    # Mix: foil_score = w_perf * percentile(mean_run_score) + w_rob * robustness_block
    mix = pd.Series({k: float(v) for k, v in foil_mix_weights.items() if float(v) > 0}, dtype=float)
    mix = mix / mix.sum()

    foils["mean_run_score__score"] = percentile_score(foils["mean_run_score"], higher_is_better=True)

    # robustness score already in 0..100 => bring to 0..1 for mixing
    rob01 = (foils["foilblock_score"] / 100.0)

    perf01 = foils["mean_run_score__score"]
    foils["foil_score"] = (
        mix.get("run_level", 0.0) * perf01 +
        mix.get("robustness", 0.0) * rob01
    ) * 100.0

    # Sort foils
    foils = foils.sort_values(["foil_score", "mean_run_score", "Plateau_pct"], ascending=[False, False, False]).reset_index(drop=True)

    return df, foils