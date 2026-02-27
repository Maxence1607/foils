import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def safe_col(df: pd.DataFrame, col: str):
    """Return series if column exists else None."""
    return df[col] if col in df.columns else None

def max_drawdown(series: pd.Series) -> float:
    """
    Max drawdown in absolute units (same unit as series):
    max(peak - trough after peak).
    """
    s = series.dropna().astype(float)
    if s.empty:
        return np.nan
    running_max = s.cummax()
    dd = running_max - s
    return float(dd.max())

def max_drawdown_pct(series: pd.Series) -> float:
    """
    Max drawdown as a fraction of the running peak (dimensionless).
    """
    s = series.dropna().astype(float)
    if s.empty:
        return np.nan
    running_max = s.cummax()
    dd_pct = (running_max - s) / running_max.replace(0, np.nan)
    return float(dd_pct.max())

def describe_series(s: pd.Series):
    """Basic stats used in the report."""
    s = s.dropna().astype(float)
    if s.empty:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def constant_value_in_run(df: pd.DataFrame, col: str):
    """Return constant value if col is constant in this run, else None."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    vals = s.unique()
    if len(vals) == 1:
        return vals[0]
    return None

# -----------------------------
# Main analysis
# -----------------------------
def analyze_run(csv_path: str, smooth_window: int = 25, show: bool = True):
    """
    Analyze a single simulation CSV.
    - smooth_window: rolling window for visualization only (in samples)
    """

    df = pd.read_csv(csv_path)
    id_file = csv_path.split('.')[0].replace('modified_Script', '').split('/')[-1]
    print(id_file)
    # If Time is missing or non-numeric, create an index-based "time"
    if "Time" not in df.columns:
        df["Time"] = np.arange(len(df))
    else:
        # Try to coerce to numeric if possible
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        if df["Time"].isna().all():
            df["Time"] = np.arange(len(df))

    # --- Parameters (constant per run)
    params_cols = [
        "Boat.Aero.Travel",
        "Boat.Keel.KeelCant",
        "Boat.Port.FoilRake",
        "Boat.Stbd.FoilRake",
        "Boat.TWS_kts",
    ]
    params = {c: constant_value_in_run(df, c) for c in params_cols if c in df.columns}

    # --- Key signals
    speed = safe_col(df, "Boat.Speed_kts")
    vmg = safe_col(df, "Boat.VMG_kts")
    helm = safe_col(df, "Boat.Helm")
    rudder = safe_col(df, "Boat.RudderDelta")
    trim = safe_col(df, "Boat.Trim")
    heel = safe_col(df, "Boat.Heel")
    leeway = safe_col(df, "Boat.Leeway")
    twa = safe_col(df, "Boat.TWA")
    power = safe_col(df, "Boat.Aero.Power")

    # --- Metrics
    metrics = {}

    if speed is not None:
        metrics["Speed_mean"] = describe_series(speed)["mean"]
        metrics["Speed_std"] = describe_series(speed)["std"]
        metrics["Speed_min"] = describe_series(speed)["min"]
        metrics["Speed_max"] = describe_series(speed)["max"]
        metrics["Speed_max_drawdown_kts"] = max_drawdown(speed)
        metrics["Speed_max_drawdown_pct"] = max_drawdown_pct(speed)

    if vmg is not None:
        metrics["VMG_mean"] = describe_series(vmg)["mean"]
        metrics["VMG_std"] = describe_series(vmg)["std"]

    if helm is not None:
        h = helm.astype(float)
        metrics["Helm_abs_mean_deg"] = float(np.nanmean(np.abs(h)))
        metrics["Helm_std_deg"] = float(np.nanstd(h, ddof=1))
        metrics["Helm_max_abs_deg"] = float(np.nanmax(np.abs(h)))

    if rudder is not None:
        r = rudder.astype(float)
        metrics["Rudder_abs_mean_deg"] = float(np.nanmean(np.abs(r)))
        metrics["Rudder_std_deg"] = float(np.nanstd(r, ddof=1))
        metrics["Rudder_max_abs_deg"] = float(np.nanmax(np.abs(r)))

    if trim is not None:
        t = trim.astype(float)
        metrics["Trim_mean_deg"] = float(np.nanmean(t))
        metrics["Trim_abs_mean_deg"] = float(np.nanmean(np.abs(t)))
        metrics["Trim_std_deg"] = float(np.nanstd(t, ddof=1))
        metrics["Trim_min_deg"] = float(np.nanmin(t))
        metrics["Trim_max_deg"] = float(np.nanmax(t))

    if heel is not None:
        he = heel.astype(float)
        metrics["Heel_mean_deg"] = float(np.nanmean(he))
        metrics["Heel_abs_mean_deg"] = float(np.nanmean(np.abs(he)))
        metrics["Heel_std_deg"] = float(np.nanstd(he, ddof=1))

    if leeway is not None:
        lw = leeway.astype(float)
        metrics["Leeway_mean_deg"] = float(np.nanmean(lw))
        metrics["Leeway_abs_mean_deg"] = float(np.nanmean(np.abs(lw)))
        metrics["Leeway_std_deg"] = float(np.nanstd(lw, ddof=1))

    if twa is not None:
        tw = pd.to_numeric(twa, errors="coerce")
        metrics["TWA_mean_deg"] = float(np.nanmean(tw))
        metrics["TWA_std_deg"] = float(np.nanstd(tw, ddof=1))
        metrics["TWA_min_deg"] = float(np.nanmin(tw))
        metrics["TWA_max_deg"] = float(np.nanmax(tw))

    if power is not None:
        p = power.astype(float)
        metrics["AeroPower_mean"] = float(np.nanmean(p))
        metrics["AeroPower_std"] = float(np.nanstd(p, ddof=1))
        # Efficiency proxy: speed per unit power (be careful with units, but useful for comparison)
        if speed is not None:
            metrics["Speed_per_AeroPower"] = float(np.nanmean(speed) / np.nanmean(p)) if np.nanmean(p) != 0 else np.nan

    # --- Print a clean report
    print("\n====================")
    print("RUN ANALYSIS")
    print("====================")
    print(f"File: {csv_path}")
    print(f"Samples: {len(df)}")

    if params:
        print("\n--- Parameters (constant per run) ---")
        for k, v in params.items():
            print(f"{k}: {v}")

    print("\n--- Metrics ---")
    # Pretty ordering
    key_order = [
        "Speed_mean", "Speed_std", "Speed_min", "Speed_max",
        "Speed_max_drawdown_kts", "Speed_max_drawdown_pct",
        "VMG_mean", "VMG_std",
        "Helm_abs_mean_deg", "Helm_std_deg", "Helm_max_abs_deg",
        "Rudder_abs_mean_deg", "Rudder_std_deg", "Rudder_max_abs_deg",
        "Trim_mean_deg", "Trim_abs_mean_deg", "Trim_std_deg", "Trim_min_deg", "Trim_max_deg",
        "Heel_mean_deg", "Heel_abs_mean_deg", "Heel_std_deg",
        "Leeway_mean_deg", "Leeway_abs_mean_deg", "Leeway_std_deg",
        "TWA_mean_deg", "TWA_std_deg", "TWA_min_deg", "TWA_max_deg",
        "AeroPower_mean", "AeroPower_std", "Speed_per_AeroPower",
    ]
    for k in key_order:
        if k in metrics:
            print(f"{k}: {metrics[k]}")

    # -----------------------------
    # Plots
    # -----------------------------
    if not show:
        return metrics, params

    time = df["Time"]

    # A) Time series: Speed + (optional) VMG
    if speed is not None:
        plt.figure()
        plt.plot(time, speed, label="Speed_kts")
        if vmg is not None:
            plt.plot(time, vmg, label="VMG_kts")
        plt.xlabel("Time")
        plt.ylabel("knots")
        plt.title("Speed / VMG over time")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_speed_vmg.png')

        # Smoothed speed
        plt.figure()
        speed_smooth = pd.Series(speed).rolling(smooth_window, min_periods=smooth_window).mean()
        plt.plot(time, speed, label="Speed_kts")
        plt.plot(time, speed_smooth, label=f"Speed rolling mean (w={smooth_window})")
        plt.xlabel("Time")
        plt.ylabel("knots")
        plt.title("Speed stability (raw vs rolling mean)")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_speed_stability.png')

        # Drawdown curve
        plt.figure()
        s = pd.Series(speed).astype(float)
        running_max = s.cummax()
        dd = running_max - s
        plt.plot(time, dd, label="Drawdown (kts)")
        plt.xlabel("Time")
        plt.ylabel("knots")
        plt.title("Speed drawdown over time")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_speed_drawdown.png')

    # B) Time series: Helm / RudderDelta
    if helm is not None:
        plt.figure()
        plt.plot(time, helm, label="Helm (deg)")
        if rudder is not None:
            plt.plot(time, rudder, label="RudderDelta (deg)")
        plt.xlabel("Time")
        plt.ylabel("degrees")
        plt.title("Steering activity over time")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_helm.png')

    # C) Time series: Trim / Heel / Leeway
    any_attitude = (trim is not None) or (heel is not None) or (leeway is not None)
    if any_attitude:
        plt.figure()
        if trim is not None:
            plt.plot(time, trim, label="Trim (deg)")
        if heel is not None:
            plt.plot(time, heel, label="Heel (deg)")
        if leeway is not None:
            plt.plot(time, leeway, label="Leeway (deg)")
        plt.xlabel("Time")
        plt.ylabel("degrees")
        plt.title("Attitude / efficiency angles over time")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_attitude.png')

    # D) TWA over time (optional)
    if twa is not None:
        plt.figure()
        plt.plot(time, pd.to_numeric(twa, errors="coerce"), label="TWA (deg)")
        plt.xlabel("Time")
        plt.ylabel("degrees")
        plt.title("TWA over time")
        plt.legend()
        #plt.show()
        plt.savefig(f'data/images/{id_file}_twa.png')

    # E) Distributions (histograms) for stability feeling
    def hist_plot(series, title, xlabel):
        plt.figure()
        plt.hist(pd.to_numeric(series, errors="coerce").dropna().astype(float), bins=40)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        #plt.show()
        plt.savefig(f'data/images/{id_file}_distribution_{title}.png')

    if speed is not None:
        hist_plot(speed, "Distribution of Speed_kts", "knots")
    if helm is not None:
        hist_plot(helm, "Distribution of Helm", "degrees")
    if trim is not None:
        hist_plot(trim, "Distribution of Trim", "degrees")
    if leeway is not None:
        hist_plot(leeway, "Distribution of Leeway", "degrees")

    return metrics, params


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    csv_path = "data/simulations/modified_Script123.csv"  # <-- adapte
    analyze_run(csv_path, smooth_window=25, show=True)