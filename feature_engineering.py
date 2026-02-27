# feature_engineering.py
# Build run-level + foil-level features from CSV simulations,
# using data/configs.json as the authoritative mapping script_id -> foil_id.
#
# Usage:
#   python feature_engineering.py --sim_dir data/simulations --config data/configs.json --out data/outputs/features.parquet
#
# Requirements:
#   pip install pandas numpy pyarrow

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Robust helpers
# -----------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def max_drawdown_pct(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    peak = np.maximum.accumulate(x)
    peak = np.where(peak == 0, np.nan, peak)
    dd = (peak - x) / peak
    return float(np.nanmax(dd))


def slope_abs(y: np.ndarray) -> float:
    y = np.abs(y[np.isfinite(y)])
    if y.size < 5:
        return np.nan
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def delta_last_first(x: np.ndarray, frac: float = 0.25) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    k = max(1, int(round(x.size * frac)))
    return float(np.mean(x[-k:]) - np.mean(x[:k]))


def q(x: np.ndarray, p: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, p))


# -----------------------------
# Mapping: configs.json -> script_id -> foil_id
# -----------------------------
def normalize_script_key(name: str) -> str:
    """
    Normalize script identifiers so that these all match:
    - "Script123.js"
    - "Script123.csv"
    - "Script123"
    - "/path/to/Script123.csv"
    """
    base = Path(name).name  # remove path
    if base.endswith(".js"):
        base = base[:-3]
    if base.endswith(".csv"):
        base = base[:-4]
    return base


def load_script_to_foil_map(config_path: str) -> dict[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    for raw_key, payload in data.items():
        # Example key: "Script[id].js"
        script_id = normalize_script_key(raw_key)

        foil = None
        try:
            foil = payload.get("Configuration", {}).get("Foil", None)
        except Exception:
            foil = None

        if foil is None:
            continue

        mapping[script_id] = str(foil)
    print(mapping)
    return mapping


# -----------------------------
# Per-run features (your blocks 1-3)
# -----------------------------
def compute_run_features(run_df: pd.DataFrame, window_frac: float = 0.25) -> dict:
    # vmg = _to_num(run_df.get("Boat.VMG_kts"))
    vmg = _to_num(run_df.get("Boat.Speed_kts"))
    spd = _to_num(run_df.get("Boat.Speed_kts"))
    heel = _to_num(run_df.get("Boat.Heel"))
    trim = _to_num(run_df.get("Boat.Trim"))
    leeway = _to_num(run_df.get("Boat.Leeway"))

    travel = _to_num(run_df.get("Boat.Aero.Travel"))
    keelcant = _to_num(run_df.get("Boat.Keel.KeelCant"))
    foilrake = _to_num(run_df.get("Boat.Port.FoilRake"))

    vmg_np = vmg.to_numpy(dtype=float)
    spd_np = spd.to_numpy(dtype=float)
    heel_np = heel.to_numpy(dtype=float)
    trim_np = trim.to_numpy(dtype=float)
    leeway_np = leeway.to_numpy(dtype=float)

    feats = {}
    # 1) Performance utile
    feats["VMG_mean"] = float(np.nanmean(vmg_np))
    feats["VMG_95p"] = q(vmg_np, 95)

    # 2) Stabilité intra-run (IQR)
    feats["IQR_Speed"] = iqr(spd_np)
    feats["IQR_Trim"] = iqr(trim_np)
    feats["IQR_Heel"] = iqr(heel_np)

    # 3) Dégradation dynamique
    feats["Delta_Speed_LastFirst"] = delta_last_first(spd_np, frac=window_frac)
    feats["MaxDrawdown_Speed_pct"] = max_drawdown_pct(spd_np)
    feats["Slope_AbsLeeway"] = slope_abs(leeway_np)

    feats["FoilRake"] = np.max(foilrake)
    feats["KeelCant"] = np.max(keelcant)
    feats["Travel"] = np.max(travel)

    # Context (optional, not used in scoring)
    if "Boat.TWS_kts" in run_df.columns:
        feats["TWS_kts_mean"] = float(np.nanmean(_to_num(run_df["Boat.TWS_kts"]).to_numpy(dtype=float)))
    if "Boat.TWA" in run_df.columns:
        feats["TWA_mean"] = float(np.nanmean(_to_num(run_df["Boat.TWA"]).to_numpy(dtype=float)))

    return feats


# -----------------------------
# Build dataset
# -----------------------------
def build_features(sim_dir: str, config_path: str, out_path: str, window_frac: float = 0.25) -> pd.DataFrame:
    sim_dir_p = Path(sim_dir)
    paths = sorted(sim_dir_p.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV found in {sim_dir}")

    script_to_foil = load_script_to_foil_map(config_path)

    rows = []
    missing_map = 0

    for p in paths:
        run_df = pd.read_csv(p)
        # print(run_df)
        script_id = normalize_script_key(p.stem)  # p.stem is without .csv, but we normalize anyway
        script_id = str(p).split('/')[-1].replace('modified_', '').replace('.csv', '')
        foil_id = script_to_foil.get(script_id)

        if foil_id is None:
            missing_map += 1
            # fallback: try normalize full filename (in case stem had extra)
            foil_id = script_to_foil.get(normalize_script_key(p.name))
        if foil_id is None:
            # If still None: keep it as "UNKNOWN" to not crash; you can filter later
            foil_id = "UNKNOWN"

        feats = compute_run_features(run_df, window_frac=window_frac)
        feats["foil_id"] = str(foil_id)
        feats["script_id"] = str(script_id)
        feats["csv_path"] = str(p)

        rows.append(feats)

    features_df = pd.DataFrame(rows)

    if missing_map:
        print(f"[WARN] {missing_map} CSV runs had no mapping in configs.json -> foil_id set to 'UNKNOWN'.")

    # -----------------------------
    # Foil-level block 4
    # -----------------------------
    # Work only on known foils for foil-level stats
    known = features_df[features_df["foil_id"] != "UNKNOWN"].copy()
    g = known.groupby("foil_id", dropna=False)

    interrun_iqr = g["VMG_mean"].apply(
        lambda s: float(np.percentile(s.dropna(), 75) - np.percentile(s.dropna(), 25)) if s.dropna().size else np.nan
    )

    interrun_mad = g["VMG_mean"].apply(
        lambda s: float(np.median(np.abs(s.dropna() - np.median(s.dropna())))) if s.dropna().size else np.nan
    )

    vmg_ref_95 = g["VMG_mean"].apply(lambda s: float(np.percentile(s.dropna(), 95)) if s.dropna().size else np.nan)

    def plateau_pct(sub: pd.DataFrame) -> float:
        ref = vmg_ref_95.loc[sub["foil_id"].iloc[0]]
        if not np.isfinite(ref):
            return np.nan
        thr = 0.95 * ref
        return float((sub["VMG_mean"] >= thr).mean())

    plateau = g.apply(plateau_pct)

    foil_level = pd.DataFrame({
        "foil_id": interrun_iqr.index.astype(str),
        "InterRun_VMG_IQR": interrun_iqr.values,
        "InterRun_VMG_MAD": interrun_mad.values,
        "Plateau_pct": plateau.values,
        "VMG_ref_95": vmg_ref_95.values,
    })

    # merge foil-level stats back into all runs (UNKNOWN stays NaN)
    features_df = features_df.merge(foil_level, on="foil_id", how="left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(features_df.columns)
    features_df.to_parquet(out_path, index=False)
    return features_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", type=str, default="data/simulations")
    ap.add_argument("--config", type=str, default="data/configs.json")
    ap.add_argument("--out", type=str, default="data/outputs/features.parquet")
    ap.add_argument("--window_frac", type=float, default=0.25)
    args = ap.parse_args()

    df = build_features(args.sim_dir, args.config, args.out, window_frac=args.window_frac)
    print(f"Saved {len(df)} runs to {args.out}")
    print("Foils:", sorted(df["foil_id"].astype(str).unique())[:20], "...")


if __name__ == "__main__":
    main()