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
# Run detail (comparaison par réglage)
# -----------------------------
with tab_run:
    st.subheader("Comparaison multi-foils pour un réglage (time series)")

    # Vérifie la présence des colonnes réglages (plan d'expériences commun)
    setting_cols = ["Travel", "KeelCant", "FoilRake"]
    missing_settings = [c for c in setting_cols if c not in scored_runs.columns]
    if missing_settings:
        st.error(f"Colonnes réglages manquantes dans features/scored_runs : {missing_settings}")
        st.stop()

    # Sélecteurs de réglage
    cA, cB, cC = st.columns(3)
    with cA:
        sel_travel = st.selectbox("Travel", sorted(scored_runs["Travel"].dropna().unique().tolist()))
    with cB:
        sel_keelcant = st.selectbox("KeelCant", sorted(scored_runs["KeelCant"].dropna().unique().tolist()))
    with cC:
        sel_foilrake = st.selectbox("FoilRake", sorted(scored_runs["FoilRake"].dropna().unique().tolist()))

    # Filtre : un run par foil (mêmes réglages)
    subset = scored_runs[
        (scored_runs["Travel"] == sel_travel)
        & (scored_runs["KeelCant"] == sel_keelcant)
        & (scored_runs["FoilRake"] == sel_foilrake)
    ].dropna(subset=["run_score"]).copy()

    if subset.empty:
        st.warning("Aucun run trouvé pour ce triplet (Travel, KeelCant, FoilRake).")
        st.stop()

    # Si jamais il y a plusieurs runs par foil sur ce triplet, on garde le meilleur run_score
    subset = subset.sort_values("run_score", ascending=False).drop_duplicates(subset=["foil_id"], keep="first")

    st.caption(f"{len(subset)} foils trouvés pour ce réglage • comparaison 'fair' (même simulation)")

    # Tableau comparatif (run-level)
    comp_cols = [
        "foil_id",
        "script_id",
        "run_score",
        "VMG_mean",
        "VMG_95p",
        "IQR_Speed",
        "IQR_Heel",
        "IQR_Trim",
        "Delta_Speed_LastFirst",
        "MaxDrawdown_Speed_pct",
        "Slope_AbsLeeway",
        "csv_path",
    ]
    comp_cols = [c for c in comp_cols if c in subset.columns]
    
# Tableau comparatif (scores normalisés 0–1 par métrique, pour ce réglage)
# Objectif: lecture immédiate forces/faiblesses (vert=bon, rouge=mauvais) sans double compter.
metric_specs = {
    "run_score": "high",
    "VMG_mean": "high",
    "VMG_95p": "high",
    "IQR_Speed": "low",
    "IQR_Heel": "low",
    "IQR_Trim": "low",
    "Delta_Speed_LastFirst": "high",
    "MaxDrawdown_Speed_pct": "low",
    "Slope_AbsLeeway": "low",
}

base_cols = ["foil_id", "script_id"]
available_metrics = [m for m in metric_specs.keys() if m in subset.columns]
score_df = subset[base_cols + available_metrics].copy()

# Convertit en scores percentile [0,1] (au sein de ce réglage), avec le bon sens par métrique
for m in available_metrics:
    s = pd.to_numeric(score_df[m], errors="coerce")
    if metric_specs[m] == "high":
        score_df[m] = s.rank(pct=True, method="average")
    else:
        score_df[m] = (-s).rank(pct=True, method="average")

score_df = score_df.sort_values("run_score", ascending=False).reset_index(drop=True)

# Affichage en heatmap (Rouge -> Vert)
heat_cols = available_metrics  # uniquement les colonnes scorées
styled = (
    score_df.style
    .format({c: "{:.2f}" for c in heat_cols})
    .background_gradient(subset=heat_cols, cmap="RdYlGn", vmin=0.0, vmax=1.0)
)
st.dataframe(styled, use_container_width=True, hide_index=True)

with st.expander("Afficher les valeurs brutes (non normalisées)", expanded=False):
    comp_cols = [
        "foil_id",
        "script_id",
        "run_score",
        "VMG_mean",
        "VMG_95p",
        "IQR_Speed",
        "IQR_Heel",
        "IQR_Trim",
        "Delta_Speed_LastFirst",
        "MaxDrawdown_Speed_pct",
        "Slope_AbsLeeway",
        "csv_path",
    ]
    comp_cols = [c for c in comp_cols if c in subset.columns]
    st.dataframe(subset[comp_cols].sort_values("run_score", ascending=False), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Courbes time-series (tous foils, même réglage)")

    # Choix du signal à tracer
    # On prend l'intersection des colonnes disponibles sur quelques CSV pour éviter les erreurs.
    sample_paths = subset_plot["csv_path"].astype(str).head(5).tolist()
    common_cols = None
    for p in sample_paths:
        try:
            dfi = load_run_csv(p)
        except Exception:
            continue
        cols = set(dfi.columns)
        common_cols = cols if common_cols is None else (common_cols & cols)
    if not common_cols:
        st.error("Impossible de déterminer les colonnes communes des CSV (lecture échouée).")
        st.stop()

    # Séries candidates (ordre de préférence)
    preferred = ["Boat.Speed_kts", "Boat.VMG_kts", "Boat.Heel", "Boat.Trim", "Boat.Leeway", "Boat.Helm"]
    candidates = [c for c in preferred if c in common_cols]
    # Ajoute le reste (tri)
    candidates += sorted([c for c in common_cols if c not in candidates])

    y_col = st.selectbox("Colonne à afficher", candidates, index=0 if candidates else 0)

# Filtre foils à afficher (sinon le graphe est vite surchargé)
foil_options = subset.sort_values("run_score", ascending=False)["foil_id"].astype(str).tolist()
default_n = min(8, len(foil_options))
default_foils = foil_options[:default_n]
selected_foils = st.multiselect(
    "Foils à afficher",
    options=foil_options,
    default=default_foils,
    help="Astuce: sélectionne seulement quelques foils (ex: top 5) pour une lecture plus claire.",
)
subset_plot = subset[subset["foil_id"].astype(str).isin([str(x) for x in selected_foils])].copy()
if subset_plot.empty:
    st.warning("Aucun foil sélectionné (ou pas de données).")
    st.stop()


    # Charge tous les runs (un par foil) et construit un long dataframe
    long_parts = []
    for _, r in subset_plot.iterrows():
        foil_id = str(r["foil_id"])
        p = str(r["csv_path"])
        try:
            dfi = load_run_csv(p)
        except Exception as e:
            st.warning(f"Lecture CSV impossible pour foil={foil_id}: {e}")
            continue

        if "Time" not in dfi.columns:
            dfi["Time"] = np.arange(len(dfi), dtype=float)

        dfi = dfi.iloc[::downsample].copy()
        dfi["value"] = pd.to_numeric(dfi[y_col], errors="coerce")
        dfi = dfi[["Time", "value"]].dropna(subset=["value"])
        dfi["foil_id"] = foil_id
        long_parts.append(dfi)

    if not long_parts:
        st.warning("Aucune série valide n'a pu être chargée pour ce réglage.")
        st.stop()

    long = pd.concat(long_parts, ignore_index=True)

    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("Time:Q", title="Time"),
            y=alt.Y("value:Q", title=y_col),
            color=alt.Color("foil_id:N", title="Foil"),
            tooltip=["foil_id", "Time", "value"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Voir l'ancien mode (un run d'un foil)", expanded=False):
        foil_for_run = st.selectbox("Foil (run)", foils, index=0, key="run_foil_old")
        g2 = (
            scored_runs[scored_runs["foil_id"].astype(str) == str(foil_for_run)]
            .dropna(subset=["run_score"])
            .sort_values("run_score", ascending=False)
        )

        run_ids = g2["script_id"].astype(str).tolist()
        sel_run = st.selectbox("Run (script_id)", run_ids, index=0, key="run_id_old")

        row = g2[g2["script_id"].astype(str) == str(sel_run)].iloc[0]
        st.metric("run_score", f"{float(row['run_score']):.2f}")

        csv_path = str(row["csv_path"])
        st.write(f"CSV: `{csv_path}`")

        df = load_run_csv(csv_path)
        if "Time" not in df.columns:
            df["Time"] = np.arange(len(df), dtype=float)

        series_default = [c for c in ["Boat.VMG_kts", "Boat.Speed_kts", "Boat.Heel", "Boat.Trim", "Boat.Leeway", "Boat.TWA"] if c in df.columns]
        series = st.multiselect("Séries", options=list(df.columns), default=series_default, key="series_old")

        plot_df = df.iloc[::downsample].copy()
        if series:
            l2 = plot_df[["Time"] + series].copy()
            for c in series:
                l2[c] = pd.to_numeric(l2[c], errors="coerce")
            l2 = l2.melt(id_vars=["Time"], var_name="series", value_name="value").dropna(subset=["value"])

            chart2 = (
                alt.Chart(l2)
                .mark_line()
                .encode(
                    x=alt.X("Time:Q"),
                    y=alt.Y("value:Q"),
                    color="series:N",
                    tooltip=["Time", "series", "value"],
                )
                .interactive()
            )
            st.altair_chart(chart2, use_container_width=True)

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