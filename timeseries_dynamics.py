"""
timeseries_dynamics.py

Small, self-contained utilities to analyze a single run (one simulation CSV already loaded
as a DataFrame) with:
- ACF (autocorrelation function)
- FFT (spectral analysis)
- Simple damping / persistence metrics (from ACF)

Designed for your use case: short runs (~120s), uniform-ish sampling, quasi-stationary oscillations.

Dependencies:
- numpy
- pandas
- matplotlib (only for plotting functions; safe to skip)
- statsmodels is NOT required (ACF computed manually)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _to_float_array(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr


def _infer_dt_seconds(time: Optional[pd.Series], n: int) -> float:
    """
    Infer sampling period dt (seconds) from a Time column if possible; otherwise assume dt=1.
    - If Time looks like seconds (monotonic numeric), dt = median diff.
    - If Time missing / unusable -> dt = 1 (unitless samples).
    """
    if time is None:
        return 1.0
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(t)) or len(t) < 3:
        return 1.0
    # Drop NaNs and check monotonicity
    t = t[~np.isnan(t)]
    if len(t) < 3:
        return 1.0
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 1.0
    # If time is not strictly increasing, still use median positive diff
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def detrend_standardize(x: np.ndarray, detrend: str = "mean") -> np.ndarray:
    """
    detrend:
      - "mean": subtract mean
      - "linear": remove best-fit line
      - "none": no detrend
    Then standardize to unit variance (if std > 0).
    """
    x = x.astype(float)
    if detrend == "mean":
        y = x - np.nanmean(x)
    elif detrend == "linear":
        idx = np.arange(len(x), dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3:
            y = x - np.nanmean(x)
        else:
            coef = np.polyfit(idx[mask], x[mask], 1)
            trend = coef[0] * idx + coef[1]
            y = x - trend
    elif detrend == "none":
        y = x.copy()
    else:
        raise ValueError("detrend must be one of {'mean','linear','none'}")

    std = np.nanstd(y, ddof=1)
    if std > 0 and np.isfinite(std):
        y = y / std
    return y


# -----------------------------
# ACF
# -----------------------------
def acf_manual(x: np.ndarray, nlags: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function up to nlags (inclusive of lag 0).

    Returns:
      lags: array of integer lags [0..nlags]
      acf:  array same length, with acf[0] = 1
    Notes:
      - x should be detrended/standardized beforehand.
      - NaNs are handled by masking finite values (pairwise). For heavy NaNs, results degrade.
    """
    x = _to_float_array(x)
    n = len(x)
    nlags = int(min(nlags, n - 1))
    lags = np.arange(nlags + 1, dtype=int)

    # If no NaNs, do fast correlation via FFT-like method? For simplicity, do direct.
    # This is fine for n~2400 and nlags~600.
    acf_vals = np.empty(nlags + 1, dtype=float)
    acf_vals[:] = np.nan

    # variance at lag 0
    x0 = x
    mask0 = np.isfinite(x0)
    if mask0.sum() < 3:
        return lags, acf_vals

    x0c = x0[mask0]
    var0 = np.dot(x0c, x0c) / len(x0c)  # since standardized, var ~ 1
    if var0 <= 0:
        return lags, acf_vals

    acf_vals[0] = 1.0
    for k in range(1, nlags + 1):
        a = x[:-k]
        b = x[k:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            acf_vals[k] = np.nan
            continue
        ak = a[m]
        bk = b[m]
        cov = np.dot(ak, bk) / len(ak)
        acf_vals[k] = cov / var0

    return lags, acf_vals


def acf_metrics(
    lags: np.ndarray,
    acf: np.ndarray,
    dt: float = 1.0,
    threshold: float = 1 / np.e,
    max_lag_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute simple persistence / damping metrics from the ACF.

    Metrics:
      - acf_lag1: ACF at lag 1
      - e_folding_time: time until |ACF| drops below threshold (default 1/e) for first time
      - integral_timescale: dt * sum_{k>=0} acf[k] for positive acf region (proxy)
      - first_zero_crossing: time until ACF crosses 0
      - dominant_acf_period: estimate dominant oscillation period from first positive peak after lag>0
    """
    lags = np.asarray(lags, dtype=int)
    acf = np.asarray(acf, dtype=float)

    # Restrict by max lag in seconds (optional)
    if max_lag_seconds is not None and dt > 0:
        max_k = int(max_lag_seconds / dt)
        max_k = max(1, min(max_k, lags.max()))
        use = lags <= max_k
        lags_u = lags[use]
        acf_u = acf[use]
    else:
        lags_u = lags
        acf_u = acf

    out: Dict[str, Any] = {}
    out["dt_seconds"] = dt
    out["acf_lag1"] = float(acf_u[1]) if len(acf_u) > 1 and np.isfinite(acf_u[1]) else np.nan

    # e-folding time: first lag where |acf| < threshold
    efold_idx = None
    for k in range(1, len(acf_u)):
        if np.isfinite(acf_u[k]) and abs(acf_u[k]) < threshold:
            efold_idx = k
            break
    out["e_folding_lag"] = int(efold_idx) if efold_idx is not None else None
    out["e_folding_time_seconds"] = float(efold_idx * dt) if efold_idx is not None else np.nan

    # first zero crossing
    zc_idx = None
    for k in range(1, len(acf_u)):
        if np.isfinite(acf_u[k]) and acf_u[k] <= 0:
            zc_idx = k
            break
    out["first_zero_crossing_lag"] = int(zc_idx) if zc_idx is not None else None
    out["first_zero_crossing_time_seconds"] = float(zc_idx * dt) if zc_idx is not None else np.nan

    # integral timescale (very standard in turbulence/time-series): sum of acf until it becomes negative
    # (or until max lag if never negative)
    if len(acf_u) >= 2 and np.isfinite(acf_u[0]):
        end = zc_idx if zc_idx is not None else len(acf_u)
        # ensure we only integrate finite values
        a = acf_u[:end]
        a = a[np.isfinite(a)]
        out["integral_timescale_seconds"] = float(dt * np.sum(a)) if len(a) > 0 else np.nan
    else:
        out["integral_timescale_seconds"] = np.nan

    # dominant period from ACF peaks:
    # Find first local maximum after lag 1 with positive ACF (indicative of oscillation period).
    dom_period = np.nan
    if len(acf_u) > 5:
        # simple peak detection
        for k in range(2, len(acf_u) - 1):
            if not (np.isfinite(acf_u[k - 1]) and np.isfinite(acf_u[k]) and np.isfinite(acf_u[k + 1])):
                continue
            if acf_u[k] > acf_u[k - 1] and acf_u[k] > acf_u[k + 1] and acf_u[k] > 0.05:
                dom_period = float(k * dt)
                break
    out["dominant_acf_period_seconds"] = dom_period
    out["dominant_acf_freq_hz"] = float(1.0 / dom_period) if np.isfinite(dom_period) and dom_period > 0 else np.nan

    return out


# -----------------------------
# FFT / Spectral analysis
# -----------------------------
def fft_spectrum(
    x: np.ndarray,
    dt: float = 1.0,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided amplitude spectrum using FFT.

    Returns:
      freqs_hz: frequencies (Hz) for one-sided spectrum (including 0)
      amp:      amplitude (not power). Relative comparisons are what matter.

    Notes:
      - x should be detrended/standardized beforehand.
      - window: 'hann' or 'none'
    """
    x = _to_float_array(x)
    n = len(x)

    # Windowing
    if window == "hann":
        w = np.hanning(n)
        xw = x * w
        # amplitude correction (rough): normalize by mean of window to keep comparable scale
        scale = np.mean(w)
        if scale > 0:
            xw = xw / scale
    elif window == "none":
        xw = x
    else:
        raise ValueError("window must be one of {'hann','none'}")

    # FFT
    fft = np.fft.rfft(xw)
    amp = np.abs(fft) / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, amp


def fft_metrics(
    freqs_hz: np.ndarray,
    amp: np.ndarray,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract simple spectral metrics:
      - dominant_freq_hz: frequency with maximum amplitude (excluding DC by default via fmin)
      - dominant_period_s
      - spectral_centroid_hz: amplitude-weighted mean frequency
      - band_energy: integrated amplitude in band (proxy)
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    amp = np.asarray(amp, dtype=float)

    mask = np.isfinite(freqs_hz) & np.isfinite(amp)
    mask &= freqs_hz >= fmin
    if fmax is not None:
        mask &= freqs_hz <= fmax

    f = freqs_hz[mask]
    a = amp[mask]
    out: Dict[str, Any] = {}

    if len(f) < 3:
        out["dominant_freq_hz"] = np.nan
        out["dominant_period_s"] = np.nan
        out["spectral_centroid_hz"] = np.nan
        out["band_energy"] = np.nan
        return out

    # dominant frequency (max amplitude)
    idx = int(np.argmax(a))
    dom_f = float(f[idx])
    out["dominant_freq_hz"] = dom_f
    out["dominant_period_s"] = float(1.0 / dom_f) if dom_f > 0 else np.nan

    # spectral centroid
    denom = float(np.sum(a))
    out["spectral_centroid_hz"] = float(np.sum(f * a) / denom) if denom > 0 else np.nan

    # band "energy" proxy (amplitude integral)
    out["band_energy"] = float(np.trapz(a, f))
    return out


# -----------------------------
# High-level wrapper
# -----------------------------
@dataclass
class DynamicsReport:
    series_name: str
    n_samples: int
    dt_seconds: float
    acf: Dict[str, Any]
    fft: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # flatten a bit for convenience
        flat = {"series_name": self.series_name, "n_samples": self.n_samples, "dt_seconds": self.dt_seconds}
        flat.update({f"acf__{k}": v for k, v in self.acf.items()})
        flat.update({f"fft__{k}": v for k, v in self.fft.items()})
        return flat


def analyze_dynamics(
    df: pd.DataFrame,
    series_col: str,
    time_col: str = "Time",
    detrend: str = "mean",
    nlags: int = 400,
    acf_threshold: float = 1 / np.e,
    max_lag_seconds: Optional[float] = None,
    fft_window: str = "hann",
    fft_fmin: float = 0.02,  # ignore DC and ultra-low drift by default
    fft_fmax: Optional[float] = None,
) -> Tuple[DynamicsReport, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Analyze one signal from one run:
      - detrend + standardize
      - ACF + damping metrics
      - FFT + spectral metrics

    Returns:
      report: DynamicsReport
      (lags, acf): arrays
      (freqs, amp): arrays
    """
    if series_col not in df.columns:
        raise KeyError(f"Column '{series_col}' not in dataframe")

    x_raw = pd.to_numeric(df[series_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_raw)
    if mask.sum() < 10:
        raise ValueError(f"Not enough finite samples in '{series_col}' to analyze")

    # Fill NaNs by linear interpolation for spectral stability (optional but helpful)
    x = pd.Series(x_raw).interpolate(limit_direction="both").to_numpy(dtype=float)

    dt = _infer_dt_seconds(df[time_col] if time_col in df.columns else None, len(x))
    xz = detrend_standardize(x, detrend=detrend)

    lags, acf_vals = acf_manual(xz, nlags=nlags)
    acf_info = acf_metrics(lags, acf_vals, dt=dt, threshold=acf_threshold, max_lag_seconds=max_lag_seconds)

    freqs, amp = fft_spectrum(xz, dt=dt, window=fft_window)
    fft_info = fft_metrics(freqs, amp, fmin=fft_fmin, fmax=fft_fmax)

    report = DynamicsReport(
        series_name=series_col,
        n_samples=len(xz),
        dt_seconds=dt,
        acf=acf_info,
        fft=fft_info,
    )
    return report, (lags, acf_vals), (freqs, amp)


# -----------------------------
# Plotting (optional)
# -----------------------------
def plot_acf(lags: np.ndarray, acf: np.ndarray, dt: float = 1.0, max_seconds: Optional[float] = None):
    import matplotlib.pyplot as plt

    t = lags * dt
    if max_seconds is not None:
        m = t <= max_seconds
        t = t[m]
        acf = acf[m]

    plt.figure()
    plt.plot(t, acf)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Lag (seconds)" if dt != 1.0 else "Lag (samples)")
    plt.ylabel("ACF")
    plt.title("Autocorrelation Function")
    plt.show()


def plot_fft(freqs_hz: np.ndarray, amp: np.ndarray, fmax: Optional[float] = None, logy: bool = False):
    import matplotlib.pyplot as plt

    f = freqs_hz
    a = amp
    if fmax is not None:
        m = f <= fmax
        f = f[m]
        a = a[m]

    plt.figure()
    plt.plot(f, a)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT Amplitude Spectrum")
    if logy:
        plt.yscale("log")
    plt.show()


def compute_run_features(
    df: pd.DataFrame,
    time_col: str = "Time",
    series_cols: Optional[List[str]] = None,
    detrend: str = "mean",
    nlags: int = 400,
    max_lag_seconds: Optional[float] = 40.0,
    acf_threshold: float = 1 / np.e,
    fft_window: str = "hann",
    fft_fmin: float = 0.02,
    fft_fmax: Optional[float] = 2.0,
    add_basic_stats: bool = True,
) -> pd.DataFrame:
    """
    Compute a single-row dataframe of time-series dynamics features for a run.

    Parameters
    ----------
    df : pd.DataFrame
        One run timeseries (e.g., ~2400 rows).
    time_col : str
        Column used to infer dt. If missing or non-numeric, dt defaults to 1.
    series_cols : list[str] or None
        Signals to analyze. Default: ["Boat.Speed_kts","Boat.Helm","Boat.Trim","Boat.Leeway"].
    detrend : str
        "mean" | "linear" | "none"
    nlags : int
        Max ACF lags (in samples).
    max_lag_seconds : float or None
        Restrict ACF metrics to a maximum lag window (in seconds). Good for comparability.
    acf_threshold : float
        Threshold for e-folding time in ACF metrics (default 1/e).
    fft_window : str
        "hann" | "none"
    fft_fmin : float
        Frequency minimum (Hz) to ignore DC / drift in FFT metrics.
    fft_fmax : float or None
        Frequency maximum (Hz) to focus on relevant oscillations.
    add_basic_stats : bool
        If True, add mean/std/min/max + drawdown for each series (when meaningful).

    Returns
    -------
    run_features : pd.DataFrame
        Single-row DF with flattened features: e.g. speed__acf__e_folding_time_seconds ...
    """
    if series_cols is None:
        series_cols = ["Boat.Speed_kts", "Boat.Helm", "Boat.Trim", "Boat.Leeway"]

    features: Dict[str, Any] = {}

    # Basic run metadata
    features["n_samples"] = int(len(df))
    features["has_time_col"] = int(time_col in df.columns)

    # Compute dynamics for each series
    for col in series_cols:
        if col not in df.columns:
            # keep explicit missing marker
            features[f"{col}__missing"] = 1
            continue

        features[f"{col}__missing"] = 0

        report, _, _ = analyze_dynamics(
            df=df,
            series_col=col,
            time_col=time_col,
            detrend=detrend,
            nlags=nlags,
            acf_threshold=acf_threshold,
            max_lag_seconds=max_lag_seconds,
            fft_window=fft_window,
            fft_fmin=fft_fmin,
            fft_fmax=fft_fmax,
        )

        # Flatten report metrics
        d = report.to_dict()
        # d contains keys like acf__..., fft__..., plus series_name / dt
        # We'll prefix with the column name for uniqueness
        for k, v in d.items():
            if k == "series_name":
                continue
            features[f"{col}__{k}"] = v

        # Optional: basic stats
        if add_basic_stats:
            s = pd.to_numeric(df[col], errors="coerce").astype(float)

            features[f"{col}__mean"] = float(np.nanmean(s))
            features[f"{col}__std"] = float(np.nanstd(s, ddof=1))
            features[f"{col}__min"] = float(np.nanmin(s))
            features[f"{col}__max"] = float(np.nanmax(s))

            # Drawdown only makes clear sense for speed-like signals; still computed generically here
            # If you prefer, restrict to Speed only.
            ss = s.dropna()
            if len(ss) >= 3:
                running_max = ss.cummax()
                dd = running_max - ss
                features[f"{col}__max_drawdown"] = float(dd.max())
                dd_pct = (running_max - ss) / running_max.replace(0, np.nan)
                features[f"{col}__max_drawdown_pct"] = float(dd_pct.max())
            else:
                features[f"{col}__max_drawdown"] = np.nan
                features[f"{col}__max_drawdown_pct"] = np.nan

    run_features = pd.DataFrame([features])
    return run_features

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: analyze one run CSV
    df = pd.read_csv("data/simulations/modified_Script10.csv")
    report_speed, (lags, acf_vals), (freqs, amp) = analyze_dynamics(df, "Boat.Speed_kts")
    print(report_speed.to_dict())
    plot_acf(lags, acf_vals, dt=report_speed.dt_seconds, max_seconds=30)
    plot_fft(freqs, amp, fmax=2.0)
    run_features = compute_run_features(df)
    print(run_features.T.head(60))  # print first 60 features (transposed)
    # pass

    # import pandas as pd
    # from timeseries_dynamics import analyze_dynamics, plot_acf, plot_fft

    # df = pd.read_csv("data/simulations/Script123.js.csv")

    # # Exemple 1: Speed
    # report, (lags, acf_vals), (freqs, amp) = analyze_dynamics(
    #     df,
    #     series_col="Boat.Speed_kts",
    #     detrend="mean",
    #     nlags=500,
    #     max_lag_seconds=40,     # pratique pour comparer sur une fenêtre commune
    #     fft_fmin=0.02,          # ignore DC
    #     fft_fmax=2.0            # limite aux oscillations < 2 Hz
    # )

    # print(report.to_dict())
    # plot_acf(lags, acf_vals, dt=report.dt_seconds, max_seconds=40)
    # plot_fft(freqs, amp, fmax=2.0)

    # # Exemple 2: Helm
    # report_h, (lags_h, acf_h), (freqs_h, amp_h) = analyze_dynamics(df, "Boat.Helm", fft_fmax=2.0)
    # print(report_h.to_dict())