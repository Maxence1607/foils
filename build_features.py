# build_features_df.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# IMPORTANT: utilise ton code existant
# - run_scoring.py contient analyze_and_score_run(...)
from run_scoring import analyze_and_score_run, RunScoreConfig


# -----------------------------
# Config
# -----------------------------
SIM_DIR = Path("data/simulations")          # dossier des CSV
JSON_PATH = Path("data/configs.json")   # <-- adapte le nom de ton json
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "Time"


# -----------------------------
# Helpers
# -----------------------------
def load_script_to_foil(json_path: Path) -> Dict[str, str]:
    """
    Expected JSON format:
    {
      "Script[id].js": {"Configuration": {"Foil": "02"}},
      ...
    }
    Returns mapping: "Script[id].js" -> "02"
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for script_name, payload in data.items():
        foil = None
        try:
            foil = payload.get("Configuration", {}).get("Foil", None)
        except Exception:
            foil = None
        if foil is not None:
            mapping[script_name] = str(foil)
    return mapping


def infer_script_name_from_csv(csv_path: Path) -> str:
    """
    Converts:
      Script123.js.csv -> Script123.js
    """
    name = csv_path.name
    if name.endswith(".csv"):
        # name = name[:-4]
        name = name.replace('modified_', '').replace('.csv', '.js')
    return name


def safe_get(d: Dict[str, Any], keys: List[str], default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_result_to_row(
    csv_path: Path,
    script_name: str,
    foil_id: Optional[str],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    result is output of analyze_and_score_run:
      - result["metrics"] dict
      - result["stationarity"] dict (per variable)
      - result["score"] dict with run_score + breakdown
    We'll create a flat row suitable for features_df.
    """
    row: Dict[str, Any] = {}

    row["csv_path"] = str(csv_path)
    row["script_name"] = script_name         # e.g. Script123.js
    row["script_id"] = script_name           # keep same naming; you can parse numeric id if you want
    row["foil_id"] = foil_id

    # --- Bring all metrics already flattened by analyze_and_score_run
    # metrics keys are like "Boat.Speed_kts__mean", "Boat.Speed_kts__acf__integral_timescale_seconds", etc.
    metrics = result.get("metrics", {})
    for k, v in metrics.items():
        row[k] = v

    # --- Add stationarity ratios as columns (for speed/helm/trim/leeway)
    st = result.get("stationarity", {})

    def add_stationarity(series_name: str, prefix: str):
        row[prefix + "mean_drift_ratio"] = safe_get(st, [series_name, "mean_drift_ratio"])
        row[prefix + "std_drift_ratio"] = safe_get(st, [series_name, "std_drift_ratio"])
        row[prefix + "slope_per_sample"] = safe_get(st, [series_name, "slope_per_sample"])

    add_stationarity("Boat.Speed_kts", "Boat.Speed_kts__stationarity__")
    add_stationarity("Boat.Helm", "Boat.Helm__stationarity__")
    add_stationarity("Boat.Trim", "Boat.Trim__stationarity__")
    add_stationarity("Boat.Leeway", "Boat.Leeway__stationarity__")

    # --- Add the run score and breakdown
    score = result.get("score", {})
    row["run_score_raw"] = safe_get(score, ["run_score"])  # from compute_run_score_from_metrics
    breakdown = safe_get(score, ["breakdown"], default={})
    if isinstance(breakdown, dict):
        for k, v in breakdown.items():
            row["score_breakdown__" + k] = v

    return row


# -----------------------------
# Main build
# -----------------------------
def build_features_df(
    sim_dir: Path,
    json_path: Path,
    time_col: str = "Time",
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    script_to_foil = load_script_to_foil(json_path)

    csv_files = sorted(sim_dir.glob("*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]

    rows: List[Dict[str, Any]] = []

    # Use a consistent scoring config (same as before)
    cfg = RunScoreConfig()

    for i, csv_path in enumerate(csv_files, start=1):
        script_name = infer_script_name_from_csv(csv_path)  # Script123.js
        foil_id = script_to_foil.get(script_name, None)

        try:
            df = pd.read_csv(csv_path)

            # Analyze (no plots when batch processing)
            result = analyze_and_score_run(
                df=df,
                time_col=time_col,
                plot=False,
                cfg=cfg,
            )

            row = flatten_result_to_row(csv_path, script_name, foil_id, result)
            rows.append(row)

        except Exception as e:
            # Keep trace of failures without breaking the batch
            rows.append({
                "csv_path": str(csv_path),
                "script_name": script_name,
                "script_id": script_name,
                "foil_id": foil_id,
                "error": str(e),
            })

        if i % 25 == 0:
            print(f"Processed {i}/{len(csv_files)} files...")

    features_df = pd.DataFrame(rows)

    # Helpful: flag rows with error
    features_df["has_error"] = features_df["error"].notna() if "error" in features_df.columns else False

    return features_df


if __name__ == "__main__":
    features_df = build_features_df(
        sim_dir=SIM_DIR,
        json_path=JSON_PATH,
        time_col=TIME_COL,
        max_files=None,  # you can set e.g. 50 to test quickly
    )

    # Save
    out_parquet = OUTPUT_DIR / "features.parquet"
    out_csv = OUTPUT_DIR / "features.csv"

    features_df.to_parquet(out_parquet, index=False)
    features_df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print("-", out_parquet)
    print("-", out_csv)

    # Quick sanity checks
    print("\nRows:", len(features_df))
    if "foil_id" in features_df.columns:
        print("Foils:", features_df["foil_id"].nunique(dropna=True))
    if "has_error" in features_df.columns:
        print("Errors:", int(features_df["has_error"].sum()))