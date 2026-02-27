# app.py
# Streamlit app for your final 4-block methodology.

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from fscoring import build_run_and_foil_scores


# -----------------------------
# Docs (simple, readable)
# -----------------------------
BLOCK_DOCS = {
    "performance_useful": {
        "title": "1️⃣ Performance utile",
        "text": "Objectif : maximiser la performance réellement utile à la course. On utilise VMG pour éviter la redondance avec Speed.",
        "metrics": {
            "VMG_mean": "Performance exploitable moyenne sur le run (↑ mieux).",
            "VMG_95p": "Plafond atteignable (95e percentile) (↑ mieux).",
        },
    },
    "stability_intrarun": {
        "title": "2️⃣ Stabilité intra-run",
        "text": "Objectif : stabilité robuste sans être sensible aux spikes. On utilise IQR (Q75−Q25) plutôt que std.",
        "metrics": {
            "IQR_Speed": "Variabilité robuste de la vitesse (↓ mieux).",
            "IQR_Trim": "Stabilité de l’assiette (↓ mieux).",
            "IQR_Heel": "Stabilité de la gîte (↓ mieux).",
        },
    },
    "dynamic_degradation": {
        "title": "3️⃣ Dégradation dynamique",
        "text": "Objectif : détecter dérive progressive et collapses, même si la moyenne paraît bonne.",
        "metrics": {
            "Delta_Speed_LastFirst": "Mean(last 25%) − Mean(first 25%) (↓ mieux si dérive/instabilité).",
            "MaxDrawdown_Speed_pct": "Risque extrême : chute peak→trough relative (↓ mieux).",
            "Slope_AbsLeeway": "Dérive latérale croissante (pente de |Leeway|) (↓ mieux).",
        },
    },
    "robustness_interrun": {
        "title": "4️⃣ Robustesse inter-simulations (niveau foil)",
        "text": "Objectif : foil performant sur une large plage de réglages (tolérance), et peu sensible aux réglages.",
        "metrics": {
            "Plateau_pct": "% de runs avec VMG_mean ≥ 95% d’une référence robuste (ref = VMG_mean 95e pct du foil) (↑ mieux).",
            "InterRun_VMG_IQR": "Dispersion robuste inter-runs de VMG_mean (↓ mieux).",
        },
    },
}


@st.cache_data(show_spinner=False)
def load_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_run_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def pareto_frontier_2d(df: pd.DataFrame, x: str, y: str, x_high=True, y_high=True) -> pd.Series:
    X = df[x].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    if not x_high:
        X = -X
    if not y_high:
        Y = -Y
    n = len(df)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (X[j] >= X[i] and Y[j] >= Y[i]) and (X[j] > X[i] or Y[j] > Y[i]):
                nd[i] = False
                break
    return pd.Series(nd, index=df.index)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Foil Selection – Paprec", layout="wide")
st.title("Foil Selection Dashboard")
st.caption("Méthode 4 blocs (run-level + foil-level) • pondération interactive • décision robuste")

with st.sidebar:
    st.header("Data")
    features_path = st.text_input("features file", "data/outputs/features.parquet")

    st.divider()
    st.header("Poids run-level (blocs 1–3)")
    w_perf = st.slider("Performance utile", 0.0, 1.0, 0.40, 0.01)
    w_stab = st.slider("Stabilité intra-run", 0.0, 1.0, 0.30, 0.01)
    w_degr = st.slider("Dégradation dynamique", 0.0, 1.0, 0.30, 0.01)

    st.divider()
    st.header("Mix foil-level")
    w_runlvl = st.slider("Importance performance moyenne (mean_run_score)", 0.0, 1.0, 0.65, 0.01)
    w_robust = st.slider("Importance robustesse inter-run (bloc 4)", 0.0, 1.0, 0.35, 0.01)

    st.divider()
    st.header("Affichage")
    topk = st.slider("Top-K runs (vue foil)", 5, 20, 10, 1)
    downsample = st.slider("Downsample run plots (N)", 1, 20, 2, 1)

# Load
try:
    features_df = load_features(features_path)
    print(features_df.columns)
except Exception as e:
    st.error(f"Impossible de charger features: {e}")
    st.stop()

required = ["foil_id", "script_id", "csv_path"]
missing = [c for c in required if c not in features_df.columns]
if missing:
    st.error(f"Colonnes manquantes dans features: {missing}")
    st.stop()

run_block_weights = {
    "performance_useful": w_perf,
    "stability_intrarun": w_stab,
    "dynamic_degradation": w_degr,
}
foil_mix_weights = {
    "run_level": w_runlvl,
    "robustness": w_robust,
}
foil_block_weights = {"robustness_interrun": 1.0}

scored_runs, scored_foils = build_run_and_foil_scores(
    features_df,
    run_block_weights=run_block_weights,
    foil_mix_weights=foil_mix_weights,
    foil_block_weights=foil_block_weights,
)

# Tabs
tab_overview, tab_method, tab_foil, tab_run, tab_decision = st.tabs(
    ["Overview", "Features & Méthodo", "Foil detail", "Run detail", "Decision"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="top")

    with c1:
        st.subheader("Classement foils (foil_score)")
        st.dataframe(
            scored_foils[["foil_id", "foil_score", "mean_run_score", "top10_mean_run_score", "top10_std_run_score", "Plateau_pct", "InterRun_VMG_IQR"]]
            .rename(columns={"top10_mean_run_score": f"top{topk}_mean_run_score", "top10_std_run_score": f"top{topk}_std_run_score"}),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.subheader("Pareto : performance vs robustesse")
        st.caption("X = mean_run_score (↑) • Y = InterRun_VMG_IQR (↓)")
        tmp = scored_foils.copy()
        tmp["pareto"] = pareto_frontier_2d(tmp, "mean_run_score", "InterRun_VMG_IQR", x_high=True, y_high=False)

        chart = (
            alt.Chart(tmp)
            .mark_circle(size=180)
            .encode(
                x=alt.X("mean_run_score:Q", title="Mean run_score (perf exploitable)"),
                y=alt.Y("InterRun_VMG_IQR:Q", title="Inter-run dispersion (IQR, lower=better)"),
                color=alt.Color("pareto:N", title="Pareto"),
                tooltip=["foil_id", "foil_score", "mean_run_score", "Plateau_pct", "InterRun_VMG_IQR"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Distribution des run_scores (tous runs)")
    hist = (
        alt.Chart(scored_runs.dropna(subset=["run_score"]))
        .mark_bar()
        .encode(
            x=alt.X("run_score:Q", bin=alt.Bin(maxbins=40), title="run_score"),
            y=alt.Y("count():Q", title="count"),
        )
    )
    st.altair_chart(hist, use_container_width=True)

# -----------------------------
# Methodology
# -----------------------------
with tab_method:
    st.subheader("Méthode 4 blocs")
    st.markdown(
        """
**Principe**
- Chaque métrique est transformée en **percentile score** (0..1) sur l’ensemble des runs/foils.
- Pour les métriques où **plus petit = mieux**, on inverse : **1 - percentile**.
- Les poids sont appliqués **par bloc**, puis répartis entre les métriques du bloc.
- Séparation stricte :
  - **run_score** = blocs 1–3 (run-level)
  - **foil_score** = mix (mean_run_score) + bloc 4 (foil-level)
"""
    )

    for key, d in BLOCK_DOCS.items():
        with st.expander(d["title"], expanded=False):
            st.write(d["text"])
            st.markdown("**Métriques :**")
            for m, desc in d["metrics"].items():
                st.write(f"- `{m}` : {desc}")

    st.divider()
    st.subheader("Poids actuels")
    st.write("Run-level :", run_block_weights)
    st.write("Foil mix :", foil_mix_weights)

# -----------------------------
# Foil detail
# -----------------------------
with tab_foil:
    st.subheader("Analyse par foil")
    foils = scored_foils["foil_id"].astype(str).tolist()
    sel = st.selectbox("Foil", foils, index=0)

    g = scored_runs[scored_runs["foil_id"].astype(str) == str(sel)].dropna(subset=["run_score"]).copy()
    g = g.sort_values("run_score", ascending=False)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Runs", int(len(g)))
    with k2:
        st.metric("Best run_score", f"{g['run_score'].max():.2f}")
    with k3:
        st.metric("Mean run_score", f"{g['run_score'].mean():.2f}")

    st.markdown("### Distribution run_score (ce foil)")
    h = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("run_score:Q", bin=alt.Bin(maxbins=25)),
            y="count():Q",
        )
    )
    st.altair_chart(h, use_container_width=True)

    st.markdown("### Top runs")
    show_cols = ["script_id", "run_score", "VMG_mean", "VMG_95p", "IQR_Speed", "MaxDrawdown_Speed_pct", "Delta_Speed_LastFirst", "Slope_AbsLeeway", "csv_path"]
    show_cols = [c for c in show_cols if c in g.columns]
    st.dataframe(g[show_cols].head(20), use_container_width=True, hide_index=True)

# -----------------------------
# Run detail
# -----------------------------
with tab_run:
    st.subheader("Analyse d’un run (time series)")
    foil_for_run = st.selectbox("Foil (run)", foils, index=0, key="run_foil")
    g2 = scored_runs[scored_runs["foil_id"].astype(str) == str(foil_for_run)].dropna(subset=["run_score"]).sort_values("run_score", ascending=False)

    run_ids = g2["script_id"].astype(str).tolist()
    sel_run = st.selectbox("Run (script_id)", run_ids, index=0)

    row = g2[g2["script_id"].astype(str) == str(sel_run)].iloc[0]
    st.metric("run_score", f"{float(row['run_score']):.2f}")

    csv_path = str(row["csv_path"])
    st.write(f"CSV: `{csv_path}`")

    df = load_run_csv(csv_path)
    # Ensure a time column for plotting
    if "Time" not in df.columns:
        df["Time"] = np.arange(len(df), dtype=float)

    series_default = [c for c in ["Boat.VMG_kts", "Boat.Speed_kts", "Boat.Heel", "Boat.Trim", "Boat.Leeway", "Boat.TWA"] if c in df.columns]
    series = st.multiselect("Séries", options=list(df.columns), default=series_default)

    plot_df = df.iloc[::downsample].copy()
    if series:
        long = plot_df[["Time"] + series].copy()
        for c in series:
            long[c] = pd.to_numeric(long[c], errors="coerce")
        long = long.melt(id_vars=["Time"], var_name="series", value_name="value").dropna(subset=["value"])

        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("Time:Q"),
                y=alt.Y("value:Q"),
                color="series:N",
                tooltip=["Time", "series", "value"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Features utilisées pour le scoring (ce run)")
    feats = ["VMG_mean", "VMG_95p", "IQR_Speed", "IQR_Trim", "IQR_Heel", "Delta_Speed_LastFirst", "MaxDrawdown_Speed_pct", "Slope_AbsLeeway"]
    present = [(f, row[f]) for f in feats if f in g2.columns]
    st.dataframe(pd.DataFrame(present, columns=["feature", "value"]), use_container_width=True, hide_index=True)

# -----------------------------
# Decision
# -----------------------------
with tab_decision:
    st.subheader("Décision & justification")
    st.markdown(
        """
**Lecture recommandée**
1) Regarder le **classement foil_score** (compromis performance exploitable + robustesse réglages).
2) Vérifier que le top 2–3 est cohérent sur le **Pareto performance vs dispersion**.
3) Ouvrir 1–2 runs “top” et 1–2 runs “mauvais” du foil candidat pour valider visuellement :
   - oscillations (heel/trim/speed)
   - drawdown
   - dérive leeway
"""
    )

    st.markdown("### Top 3 foils (avec les poids actuels)")
    st.dataframe(scored_foils.head(3)[["foil_id", "foil_score", "mean_run_score", "Plateau_pct", "InterRun_VMG_IQR"]], use_container_width=True, hide_index=True)

    st.markdown("### Pourquoi le classement change quand on bouge les poids")
    st.write(
        "- Si tu augmentes **Performance utile**, tu favorises les foils avec VMG_mean/VMG_95p élevés.\n"
        "- Si tu augmentes **Stabilité intra-run**, tu favorises les runs “smooth” (IQR bas).\n"
        "- Si tu augmentes **Dégradation dynamique**, tu pénalises les runs qui s’effondrent ou dérivent.\n"
        "- Si tu augmentes **Robustesse**, tu favorises les foils qui restent proches du plafond sur beaucoup de réglages."
    )