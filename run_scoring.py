"""
run_scoring.py

End-to-end analysis for ONE simulation run:
- stationarity checks (mean/variance drift across segments)
- ACF + damping metrics
- FFT + spectral metrics
- a single "run_score" (performance + stability + control)

Assumes you already have timeseries_dynamics.py available with:
- analyze_dynamics(...)
- plot_acf(...)
- plot_fft(...)

Dependencies: numpy, pandas, matplotlib (for plotting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeseries_dynamics import analyze_dynamics, plot_acf, plot_fft


# -----------------------------
# Stationarity utilities
# -----------------------------
def _split_indices(n: int, n_segments: int = 3) -> List[Tuple[int, int]]:
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_segments)]


def stationarity_report(
    df: pd.DataFrame,
    col: str,
    n_segments: int = 3,
) -> Dict[str, Any]:
    """
    Simple and defensible stationarity diagnostics:
    - split series into segments
    - compute mean/std per segment
    - compute drift ratios between first and last segment
    """
    if col not in df.columns:
        return {"missing": 1}

    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    n = len(s)
    segs = _split_indices(n, n_segments=n_segments)

    seg_stats = []
    for (a, b) in segs:
        seg = s.iloc[a:b].dropna()
        if len(seg) < 10:
            seg_stats.append({"mean": np.nan, "std": np.nan})
        else:
            seg_stats.append({"mean": float(seg.mean()), "std": float(seg.std(ddof=1))})

    m0, mL = seg_stats[0]["mean"], seg_stats[-1]["mean"]
    s0, sL = seg_stats[0]["std"], seg_stats[-1]["std"]
    mG = float(s.dropna().mean()) if s.dropna().size else np.nan
    sG = float(s.dropna().std(ddof=1)) if s.dropna().size else np.nan

    # drift ratios (dimensionless)
    mean_drift_ratio = (abs(mL - m0) / abs(mG)) if (np.isfinite(m0) and np.isfinite(mL) and np.isfinite(mG) and abs(mG) > 1e-9) else np.nan
    std_drift_ratio = (abs(sL - s0) / (sG if (np.isfinite(sG) and sG > 1e-9) else np.nan)) if (np.isfinite(s0) and np.isfinite(sL)) else np.nan

    # linear trend slope (per second if Time available)
    # (robust enough; not a formal test)
    idx = np.arange(n, dtype=float)
    mask = np.isfinite(s.to_numpy(dtype=float))
    slope = np.nan
    if mask.sum() >= 20:
        coef = np.polyfit(idx[mask], s.to_numpy(dtype=float)[mask], 1)
        slope = float(coef[0])  # per sample; we can convert to per second later if dt known

    return {
        "missing": 0,
        "n": n,
        "segment_stats": seg_stats,
        "mean_global": mG,
        "std_global": sG,
        "mean_drift_ratio": mean_drift_ratio,
        "std_drift_ratio": std_drift_ratio,
        "slope_per_sample": slope,
    }


# -----------------------------
# Scoring utilities
# -----------------------------
def _percentile_rank(value: float, ref: np.ndarray, higher_is_better: bool = True) -> float:
    """Return 0..1 percentile rank (robust to NaNs)."""
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or ref.size == 0:
        return np.nan
    if higher_is_better:
        return float(np.mean(ref <= value))
    else:
        # lower is better
        return float(np.mean(ref >= value))


@dataclass
class RunScoreConfig:
    """
    Scoring philosophy (for ONE run):
    - reward performance (speed_mean)
    - penalize instability (speed_std, speed_drawdown_pct)
    - penalize control effort (helm_abs_mean, leeway_abs_mean)
    - penalize non-stationarity (mean/std drift ratios)

    NOTE: For ranking runs, it’s better to compute these metrics for ALL runs,
          then normalize across the dataset. Here we provide a single-run score
          that is still meaningful within the run, but will be strongest when
          you compare scores across runs computed the same way.
    """
    # weights (positive adds, negative subtracts)
    w_speed: float = 1.00
    w_speed_std: float = 0.40
    w_speed_dd: float = 0.60
    w_helm: float = 0.25
    w_leeway: float = 0.15
    w_stationarity: float = 0.60

    # stationarity thresholds (ratios)
    # mean drift ratio: |mean_last - mean_first| / |mean_global|
    mean_drift_ok: float = 0.01   # 1%
    std_drift_ok: float = 0.15    # 15%

    # FFT/ACF optional penalties (persistence too high = potentially "oscillatory")
    w_persistence: float = 0.15
    persistence_ref_seconds: float = 10.0  # integral timescale target; above this penalize mildly


def compute_run_score_from_metrics(
    metrics: Dict[str, Any],
    station: Dict[str, Any],
    cfg: RunScoreConfig,
) -> Dict[str, Any]:
    """
    Compute a single scalar score + breakdown.
    This score is intentionally simple and explainable.
    """
    # required keys
    speed_mean = metrics.get("Boat.Speed_kts__mean", np.nan)
    speed_std = metrics.get("Boat.Speed_kts__std", np.nan)
    speed_dd_pct = metrics.get("Boat.Speed_kts__max_drawdown_pct", np.nan)

    helm_abs_mean = metrics.get("Boat.Helm__mean", np.nan)  # NOTE: could be signed mean, better use abs mean if you store it
    # If you computed abs mean separately, prefer that:
    helm_abs_mean = metrics.get("Boat.Helm__abs_mean", helm_abs_mean)
    leeway_abs_mean = metrics.get("Boat.Leeway__abs_mean", metrics.get("Boat.Leeway__mean", np.nan))

    # ACF integral timescale on speed
    tint = metrics.get("Boat.Speed_kts__acf__integral_timescale_seconds", np.nan)

    # stationarity
    mean_drift = station.get("mean_drift_ratio", np.nan)
    std_drift = station.get("std_drift_ratio", np.nan)

    # Stationarity penalty: 0 if within thresholds; grows if above
    stat_pen = 0.0
    if np.isfinite(mean_drift):
        stat_pen += max(0.0, (mean_drift - cfg.mean_drift_ok) / max(cfg.mean_drift_ok, 1e-6))
    else:
        stat_pen += 1.0
    if np.isfinite(std_drift):
        stat_pen += max(0.0, (std_drift - cfg.std_drift_ok) / max(cfg.std_drift_ok, 1e-6))
    else:
        stat_pen += 1.0
    stat_pen = float(stat_pen)

    # Persistence penalty: only mild; above reference, penalize
    pers_pen = 0.0
    if np.isfinite(tint):
        pers_pen = max(0.0, (tint - cfg.persistence_ref_seconds) / max(cfg.persistence_ref_seconds, 1e-6))
    else:
        pers_pen = 0.0

    # Core score (higher better)
    # speed_std and drawdown are penalties -> subtract
    score = (
        cfg.w_speed * (speed_mean if np.isfinite(speed_mean) else 0.0)
        - cfg.w_speed_std * (speed_std if np.isfinite(speed_std) else 0.0)
        - cfg.w_speed_dd * ((speed_dd_pct * 100.0) if np.isfinite(speed_dd_pct) else 0.0)  # pct -> points
        - cfg.w_helm * (abs(helm_abs_mean) if np.isfinite(helm_abs_mean) else 0.0)
        - cfg.w_leeway * (abs(leeway_abs_mean) if np.isfinite(leeway_abs_mean) else 0.0)
        - cfg.w_stationarity * stat_pen
        - cfg.w_persistence * pers_pen
    )

    return {
        "run_score": float(score),
        "breakdown": {
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "speed_drawdown_pct": speed_dd_pct,
            "helm_abs_mean_used": helm_abs_mean,
            "leeway_abs_mean_used": leeway_abs_mean,
            "speed_integral_timescale_s": tint,
            "stationarity_mean_drift_ratio": mean_drift,
            "stationarity_std_drift_ratio": std_drift,
            "stationarity_penalty": stat_pen,
            "persistence_penalty": float(pers_pen),
        },
    }


# -----------------------------
# End-to-end analysis for one run
# -----------------------------
def analyze_and_score_run(
    df: pd.DataFrame,
    time_col: str = "Time",
    series_for_dynamics: Optional[List[str]] = None,
    cfg: Optional[RunScoreConfig] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline for one run:
    - basic stats + drawdown for Speed/Helm/Trim/Leeway
    - stationarity checks (Speed + Helm + Trim + Leeway)
    - ACF+FFT for each of Speed/Helm/Trim/Leeway
    - score computed mainly from Speed + stability + control + stationarity

    Returns a dict you can log or store.
    """
    if cfg is None:
        cfg = RunScoreConfig()

    if series_for_dynamics is None:
        series_for_dynamics = ["Boat.Speed_kts", "Boat.Helm", "Boat.Trim", "Boat.Leeway"]

    out: Dict[str, Any] = {}

    # --- Build a metrics dict like your wrapper output (minimal for scoring + reporting)
    metrics: Dict[str, Any] = {}

    # Basic stats + drawdown
    for col in series_for_dynamics:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        metrics[f"{col}__mean"] = float(np.nanmean(s))
        metrics[f"{col}__std"] = float(np.nanstd(s, ddof=1))
        metrics[f"{col}__min"] = float(np.nanmin(s))
        metrics[f"{col}__max"] = float(np.nanmax(s))

        ss = s.dropna()
        if len(ss) >= 3:
            running_max = ss.cummax()
            dd = running_max - ss
            metrics[f"{col}__max_drawdown"] = float(dd.max())
            dd_pct = (running_max - ss) / running_max.replace(0, np.nan)
            metrics[f"{col}__max_drawdown_pct"] = float(dd_pct.max())
        else:
            metrics[f"{col}__max_drawdown"] = np.nan
            metrics[f"{col}__max_drawdown_pct"] = np.nan

        # Also store abs mean for angular signals (useful for Helm/Trim/Leeway)
        if col in ["Boat.Helm", "Boat.Trim", "Boat.Leeway"]:
            metrics[f"{col}__abs_mean"] = float(np.nanmean(np.abs(ss))) if len(ss) else np.nan

    # --- Stationarity reports (keep full detail)
    stationarity: Dict[str, Any] = {}
    for col in series_for_dynamics:
        stationarity[col] = stationarity_report(df, col, n_segments=3)

    # We'll use Speed stationarity for the score (you can change later)
    speed_station = stationarity.get("Boat.Speed_kts", {"missing": 1})

    # --- Dynamics (ACF + FFT) for each series
    dynamics: Dict[str, Any] = {}
    acf_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    fft_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for col in series_for_dynamics:
        if col not in df.columns:
            continue

        report, (lags, acf_vals), (freqs, amp) = analyze_dynamics(
            df=df,
            series_col=col,
            time_col=time_col,
            detrend="mean",
            nlags=500,
            max_lag_seconds=40.0,
            fft_window="hann",
            fft_fmin=0.02,
            fft_fmax=2.0,
        )
        dynamics[col] = report.to_dict()
        acf_arrays[col] = (lags, acf_vals)
        fft_arrays[col] = (freqs, amp)

        # Add key dynamics fields into metrics for easy scoring/printing
        # (prefix with original column name)
        for k, v in report.to_dict().items():
            if k == "series_name":
                continue
            metrics[f"{col}__{k}"] = v

    # --- Score
    score = compute_run_score_from_metrics(metrics, speed_station, cfg)

    # --- Optional plots
    if plot:
        t = df[time_col] if time_col in df.columns else pd.Series(np.arange(len(df)))
        t = pd.to_numeric(t, errors="coerce")
        if t.isna().all():
            t = pd.Series(np.arange(len(df)), name="Time")

        # Time series overview
        plt.figure()
        if "Boat.Speed_kts" in df.columns:
            plt.plot(t, pd.to_numeric(df["Boat.Speed_kts"], errors="coerce"), label="Speed (kts)")
        if "Boat.Helm" in df.columns:
            plt.plot(t, pd.to_numeric(df["Boat.Helm"], errors="coerce"), label="Helm (deg)")
        plt.xlabel("Time")
        plt.title("Run overview: Speed & Helm")
        plt.legend()
        plt.show()

        plt.figure()
        if "Boat.Trim" in df.columns:
            plt.plot(t, pd.to_numeric(df["Boat.Trim"], errors="coerce"), label="Trim (deg)")
        if "Boat.Leeway" in df.columns:
            plt.plot(t, pd.to_numeric(df["Boat.Leeway"], errors="coerce"), label="Leeway (deg)")
        plt.xlabel("Time")
        plt.title("Run overview: Trim & Leeway")
        plt.legend()
        plt.show()

        # Stationarity segment stats plot for Speed
        if "Boat.Speed_kts" in stationarity and stationarity["Boat.Speed_kts"].get("missing") == 0:
            segs = stationarity["Boat.Speed_kts"]["segment_stats"]
            means = [d["mean"] for d in segs]
            stds = [d["std"] for d in segs]
            plt.figure()
            plt.plot(range(1, len(means) + 1), means, marker="o", label="segment mean")
            plt.plot(range(1, len(stds) + 1), stds, marker="o", label="segment std")
            plt.xticks(range(1, len(means) + 1))
            plt.xlabel("Segment index (1..3)")
            plt.title("Stationarity check (Speed): mean & std by segment")
            plt.legend()
            plt.show()

        # ACF/FFT plots for Speed and Helm (most informative)
        for col in ["Boat.Speed_kts", "Boat.Helm"]:
            if col in acf_arrays:
                lags, acf_vals = acf_arrays[col]
                dt = metrics.get(f"{col}__dt_seconds", metrics.get(f"{col}__acf__dt_seconds", 1.0))
                plot_acf(lags, acf_vals, dt=float(dt), max_seconds=40)
            if col in fft_arrays:
                freqs, amp = fft_arrays[col]
                plot_fft(freqs, amp, fmax=2.0, logy=False)

    out["metrics"] = metrics
    out["stationarity"] = stationarity
    out["dynamics"] = dynamics
    out["score"] = score
    out["score_config"] = cfg.__dict__
    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/simulations/modified_Script10.csv")
    result = analyze_and_score_run(df, time_col="Time", plot=True)

    print("\nRUN SCORE:", result["score"]["run_score"])
    print("\nSCORE BREAKDOWN:")
    for k, v in result["score"]["breakdown"].items():
        print(f"- {k}: {v}")

    print("\nSTATIONARITY (Speed):")
    print(result["stationarity"]["Boat.Speed_kts"])