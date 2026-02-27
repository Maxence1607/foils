# app.py
# Streamlit app — run-level scoring only (no foil-level scoring)

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
        "title": "Performance utile",
        "text": "Objectif : maximiser la performance réellement utile à la course. On utilise VMG pour éviter la redondance avec Speed.",
        "metrics": {
            "VMG_mean": "Performance exploitable moyenne sur le run (↑ mieux).",
            "VMG_95p": "Plafond atteignable (95e percentile) (↑ mieux).",
        },
    },
    "stability_intrarun": {
        "title": "Stabilité intra-run",
        "text": "Objectif : stabilité robuste sans être sensible aux spikes. On utilise IQR (Q75−Q25) plutôt que std.",
        "metrics": {
            "IQR_Speed": "Variabilité robuste de la vitesse (↓ mieux).",
            "IQR_Trim": "Stabilité de l’assiette (↓ mieux).",
            "IQR_Heel": "Stabilité de la gîte (↓ mieux).",
        },
    },
    "dynamic_degradation": {
        "title": "Dégradation dynamique",
        "text": "Objectif : détecter dérive progressive et collapses, même si la moyenne paraît bonne.",
        "metrics": {
            # NB: si Delta = last - first, alors ↑ mieux. Ajuste si ta définition est inverse.
            "Delta_Speed_LastFirst": "Mean(last 25%) − Mean(first 25%) (↑ mieux).",
            "MaxDrawdown_Speed_pct": "Risque extrême : chute peak→trough relative (↓ mieux).",
            "Slope_AbsLeeway": "Dérive latérale croissante (pente de |Leeway|) (↓ mieux).",
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
    """Naive non-dominated set in 2D. Returns boolean mask series."""
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


def build_foil_table_from_runs(
    scored_runs: pd.DataFrame,
    topk: int = 10,
    good_thr: float = 0.75,
    bad_thr: float = 0.25,
) -> pd.DataFrame:
    """Aggregate foil-level summary WITHOUT using foil-level scoring."""
    df = scored_runs.dropna(subset=["run_score"]).copy()
    df["foil_id"] = df["foil_id"].astype(str)

    # Global percentile of run_score on all runs (N*60)
    if "run_score__score" not in df.columns:
        df["run_score__score"] = pd.to_numeric(df["run_score"], errors="coerce").rank(pct=True, method="average")

    def agg_one(g: pd.DataFrame) -> pd.Series:
        g_sorted = g.sort_values("run_score", ascending=False)
        top = g_sorted.head(topk)
        return pd.Series(
            {
                "n_runs": int(len(g_sorted)),
                "best_run_score": float(g_sorted["run_score"].max()),
                f"top{topk}_mean_run_score": float(top["run_score"].mean()) if len(top) else np.nan,
                f"top{topk}_std_run_score": float(top["run_score"].std(ddof=0)) if len(top) else np.nan,
                "good_runs": int((g_sorted["run_score__score"] >= good_thr).sum()),
                "bad_runs": int((g_sorted["run_score__score"] <= bad_thr).sum()),
            }
        )

    out = df.groupby("foil_id", as_index=False).apply(agg_one)
    # groupby+apply produces MultiIndex-like; normalize:
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index(level=0, drop=True).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    # Sort by mean_run_score desc by default (you can change to good_runs desc)
    out = out.sort_values("best_run_score", ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Foil Selection – Paprec", layout="wide")
st.title("Foil Selection Dashboard")
# st.caption("Scoring run-level uniquement (features-level) • comparaison multi-foils par réglage • lecture décisionnelle")

with st.sidebar:
    st.header("Data")
    features_path = st.text_input("features file", "data/outputs/features.parquet")

    st.divider()
    st.header("Poids run-level (blocs 1–3)")
    w_perf = st.slider("Performance utile", 0.0, 1.0, 0.40, 0.01)
    w_stab = st.slider("Stabilité intra-run", 0.0, 1.0, 0.30, 0.01)
    w_degr = st.slider("Dégradation dynamique", 0.0, 1.0, 0.30, 0.01)

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

# We still call build_run_and_foil_scores to compute run_score + metric __score columns,
# but we IGNORE foil-level scoring and build our own foil table from runs.
scored_runs, _unused_scored_foils = build_run_and_foil_scores(
    features_df,
    run_block_weights=run_block_weights,
    foil_mix_weights={"run_level": 1.0, "robustness": 0.0},
    foil_block_weights={"robustness_interrun": 0.0},
)

# Global percentile of run_score for coloring/thresholds (N*60)
scored_runs["run_score__score"] = (
    pd.to_numeric(scored_runs["run_score"], errors="coerce")
    .rank(pct=True, method="average")
)

scored_runs['run_score'] = np.round(scored_runs['run_score'], 2)
# Foil summary table (no foil_score)
scored_foils = build_foil_table_from_runs(scored_runs, topk=topk, good_thr=0.75, bad_thr=0.25)

# Tabs
tab_method, tab_run, tab_foil, tab_overview = st.tabs(
    ["Features & Méthodo", "Run detail", "Foil detail", "Overview"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    c1, c2 = st.columns([1.2, 1.0], vertical_alignment="top")

    with c1:
        st.subheader("Classement foils")
        show = scored_foils[
            ["foil_id", "best_run_score", f"top{topk}_mean_run_score", f"top{topk}_std_run_score", "good_runs", "bad_runs"]
        ].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Pareto : good_runs vs bad_runs")
        st.caption("X = good_runs (↑) • Y = bad_runs (↓) — plus de runs “bons”, moins de runs “mauvais”")
        tmp = scored_foils.copy()
        tmp["pareto"] = pareto_frontier_2d(tmp, "good_runs", "bad_runs", x_high=True, y_high=False)

        chart = (
            alt.Chart(tmp)
            .mark_circle(size=180)
            .encode(
                x=alt.X("good_runs:Q", title="good_runs (run_score ≥ 75e pct global)"),
                y=alt.Y("bad_runs:Q", title="bad_runs (run_score ≤ 25e pct global)"),
                color=alt.Color("pareto:N", title="Pareto"),
                tooltip=["foil_id", "n_runs", "best_run_score", "good_runs", "bad_runs"],
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
# Methodology / “Synthèse” text
# -----------------------------
with tab_method:
    st.subheader("Synthèse & méthodologie")
    # st.caption("Chaque run reçoit un run_score normalisé (via percentiles sur N×60). Les foils sont ensuite comparés via l’agrégation des runs.")

    st.markdown(
        """
        L’objectif est d’identifier le foil offrant le meilleur compromis entre performance exploitable et robustesse aux réglages pour une exploitation opérationnelle fiable.
### Principe
- Chaque métrique est convertie en **score percentile global** (0..1) calculé sur l’ensemble des runs (**N×60**).
- Pour les métriques où “plus petit = mieux”, on inverse : **1 − percentile**.
- **run_score** (run-level) = combinaison pondérée de 3 blocs :
  1) Performance utile (VMG)  
  2) Stabilité intra-run (variabilité robuste)  
  3) Dégradation dynamique (dérive / collapses)
""")

    for key, d in BLOCK_DOCS.items():
        with st.expander(d["title"], expanded=False):
            st.write(d["text"])
            st.markdown("**Métriques :**")
            for m, desc in d["metrics"].items():
                st.write(f"- `{m}` : {desc}")

    st.markdown(
"""
### Classement des foils
- On agrège ensuite **les runs** par foil (best_run_score, topK, etc.).
- Deux compteurs robustes permettent une lecture immédiate :
  - **good_runs** : nombre de runs du foil au-dessus du **75e percentile global**
  - **bad_runs** : nombre de runs du foil en-dessous du **25e percentile global**
- Le Pareto utilise **good_runs (↑)** vs **bad_runs (↓)**.
"""
    )

    
    
    st.subheader("Résultats clés & recommandation (auto)")

    # Assure-toi que les colonnes existent (sinon adapte les noms)
    needed_cols = ["foil_id", "best_run_score", "top10_mean_run_score", "top10_std_run_score", "good_runs", "bad_runs"]
    missing_cols = [c for c in needed_cols if c not in scored_foils.columns]
    if missing_cols:
        st.warning(
            "Impossible de générer la synthèse automatique : colonnes manquantes dans scored_foils.\n"
            f"Manquantes : {missing_cols}\n"
            "➡️ Vérifie le nom de tes colonnes (ex: best_run_score vs best_run_score)."
        )
    else:
        # Top 3 foils
        top3 = scored_foils.copy().head(3).reset_index(drop=True)

        # Choix du foil recommandé :
        # 1) top10_mean_run_score (plus haut = mieux)
        # 2) best_run_score (plus haut = mieux)
        # 3) bad_runs (plus bas = mieux)
        top3_rank = top3.sort_values(
            ["top10_mean_run_score", "best_run_score", "bad_runs"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        recommended = top3_rank.iloc[0]

        # Helpers
        def f2(x):  # format float 2 decimals
            try:
                return f"{float(x):.2f}"
            except Exception:
                return str(x)

        def f0(x):  # format int
            try:
                return f"{int(x)}"
            except Exception:
                return str(x)

        # Data extraction
        r = {row["foil_id"]: row for _, row in top3.iterrows()}

        # On va aussi identifier le foil "plus risqué" vs "plus stable" dans le top3
        # - plus stable = std top10 la plus faible
        most_stable = top3.sort_values("top10_std_run_score", ascending=True).iloc[0]
        # - plus polarisé/risqué = bad_runs élevé (dans le top3)
        most_risky = top3.sort_values("bad_runs", ascending=False).iloc[0]

        st.markdown("### Résultats clés")

        # 1) Plafond de perf
        best_by_peak = top3.sort_values("best_run_score", ascending=False).iloc[0]
        st.markdown(
            f"- **Plafond de performance** : les meilleurs runs sont proches. "
            f"Le meilleur plafond parmi le Top 3 est **Foil {best_by_peak['foil_id']}** "
            f"avec un *best_run_score* de **{f2(best_by_peak['best_run_score'])}**."
        )

        # 2) Perf moyenne sur top10
        best_by_top10 = top3.sort_values("top10_mean_run_score", ascending=False).iloc[0]
        st.markdown(
            f"- **Performance moyenne sur les 10 meilleurs réglages** : "
            f"**Foil {best_by_top10['foil_id']}** est en tête avec un *top10_mean_run_score* de "
            f"**{f2(best_by_top10['top10_mean_run_score'])}**."
        )

        # 3) Stabilité dans les meilleurs réglages
        st.markdown(
            f"- **Stabilité des meilleures performances** : "
            f"**Foil {most_stable['foil_id']}** est le plus régulier dans son Top 10 "
            f"(*top10_std_run_score* = **{f2(most_stable['top10_std_run_score'])}**)."
        )

        # 4) Robustesse (good/bad)
        st.markdown(
            f"- **Robustesse globale (distribution des runs)** : "
            f"le nombre de *good_runs* (≥ 75e percentile global) et *bad_runs* (≤ 25e) "
            f"met en évidence des profils différents. "
            f"Dans ce Top 3, **Foil {most_risky['foil_id']}** présente le niveau de risque le plus élevé "
            f"au sens de runs défavorables (*bad_runs* = **{f0(most_risky['bad_runs'])}**)."
        )

        # mini-table recap
        st.markdown("**Récapitulatif Top 3**")
        st.dataframe(
            top3[["foil_id", "best_run_score", "top10_mean_run_score", "top10_std_run_score", "good_runs", "bad_runs"]],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("### Recommandation")

        st.markdown(
            f"**Foil recommandé : Foil {recommended['foil_id']}**\n\n"
            f"Sur la base des indicateurs disponibles, **Foil {recommended['foil_id']}** présente le meilleur compromis "
            f"entre performance exploitable et maîtrise du risque :\n"
            f"- *top10_mean_run_score* = **{f2(recommended['top10_mean_run_score'])}** (niveau de performance sur configurations optimales)\n"
            f"- *best_run_score* = **{f2(recommended['best_run_score'])}** (plafond atteignable)\n"
            f"- *top10_std_run_score* = **{f2(recommended['top10_std_run_score'])}** (régularité dans le Top 10)\n"
            f"- *good_runs* = **{f0(recommended['good_runs'])}** / *bad_runs* = **{f0(recommended['bad_runs'])}** (profil de robustesse)\n"
        )

        # Alternative (si on veut absolument maximiser good_runs)
        best_good = top3.sort_values(["good_runs", "bad_runs"], ascending=[False, True]).iloc[0]
        if best_good["foil_id"] != recommended["foil_id"]:
            st.markdown(
                f"**Alternative “potentiel élevé” : Foil {best_good['foil_id']}**\n\n"
                f"Si la stratégie consiste à maximiser le nombre de configurations très performantes, "
                f"**Foil {best_good['foil_id']}** est une alternative crédible (good_runs = **{f0(best_good['good_runs'])}**). "
                f"En contrepartie, surveiller le niveau de *bad_runs* (**{f0(best_good['bad_runs'])}**) qui peut indiquer "
                f"une sensibilité accrue aux réglages."
            )

        # Validation recommandée
        st.markdown(
            "### Validation recommandée\n"
            "- Confirmer la hiérarchie sur 2–3 réglages représentatifs + 1–2 réglages “difficiles”.\n"
            "- Vérifier sur les time-series (onglet Run detail) : oscillations (heel/trim/speed), drawdown, dérive leeway.\n"
            "- Vérifier la tolérance aux réglages voisins (robustesse)."
        )

    # st.divider()
    # st.subheader("Poids actuels")
    # st.write("Run-level (blocs 1–3) :", run_block_weights)

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

    st.markdown("### Distribution run_score")
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
    show_cols = [
        "KeelCant", "Travel", "FoilRake",
        "script_id", "run_score",
        "VMG_mean", "VMG_95p",
        "IQR_Speed", "MaxDrawdown_Speed_pct", "Delta_Speed_LastFirst", "Slope_AbsLeeway",
    ]
    show_cols = [c for c in show_cols if c in g.columns]
    st.dataframe(g[show_cols].head(20), use_container_width=True, hide_index=True)

# -----------------------------
# Run detail (comparaison par réglage)
# -----------------------------
with tab_run:
    st.subheader("Comparaison multi-foils pour un réglage")

    # Vérifie la présence des colonnes réglages
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

    subset = scored_runs[
        (scored_runs["Travel"] == sel_travel)
        & (scored_runs["KeelCant"] == sel_keelcant)
        & (scored_runs["FoilRake"] == sel_foilrake)
    ].dropna(subset=["run_score"]).copy()

    if subset.empty:
        st.warning("Aucun run trouvé pour ce triplet (Travel, KeelCant, FoilRake).")
        st.stop()

    # Un run par foil (si doublons, garde le meilleur run_score)
    subset = subset.sort_values("run_score", ascending=False).drop_duplicates(subset=["foil_id"], keep="first")
    st.caption(f"{len(subset)} foils trouvés pour ce réglage • comparaison 'fair'")

    # Heatmap scores (GLOBAL 0..1)
        # Heatmap scores (GLOBAL 0..1) — FIX: séparer run_score brut et run_score score
    metric_names = [
        "run_score", "VMG_mean", "VMG_95p",
        "IQR_Speed", "IQR_Heel", "IQR_Trim",
        "Delta_Speed_LastFirst", "MaxDrawdown_Speed_pct", "Slope_AbsLeeway",
    ]
    base_cols = ["foil_id", "script_id"]

    # 1) Construire un DF avec run_score brut pour tri
    score_df = subset[base_cols + ["run_score"]].copy().rename(columns={"run_score": "run_score_raw"})

    # 2) Ajouter la colonne run_score (score 0..1) pour la heatmap
    if "run_score__score" in subset.columns:
        score_df["run_score"] = subset["run_score__score"].values
    else:
        # fallback : calcule sur le subset (moins bien), mais évite crash
        score_df["run_score"] = pd.to_numeric(subset["run_score_raw"], errors="coerce").rank(pct=True, method="average")

    # 3) Ajouter les scores des métriques (colonnes __score déjà globales)
    for m in metric_names:
        if m == "run_score":
            continue
        sc = f"{m}__score"
        if sc in subset.columns:
            score_df[m] = subset[sc].values

    # 4) Colonnes à colorer
    heat_cols = [c for c in metric_names if c in score_df.columns]

    # 5) Trier sur run_score brut (pas le percentile)
    score_df = score_df.sort_values("run_score_raw", ascending=False).reset_index(drop=True)

    # 6) Affichage heatmap (0..1)
    styled = (
        score_df.style
        .format({c: "{:.2f}" for c in heat_cols})
        .background_gradient(subset=heat_cols, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("Afficher les valeurs brutes (non normalisées)", expanded=False):
        comp_cols = [
            "foil_id", "script_id", "run_score",
            "VMG_mean", "VMG_95p",
            "IQR_Speed", "IQR_Heel", "IQR_Trim",
            "Delta_Speed_LastFirst", "MaxDrawdown_Speed_pct", "Slope_AbsLeeway",
        ]
        comp_cols = [c for c in comp_cols if c in subset.columns]
        st.dataframe(
            subset[comp_cols].sort_values("run_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("### Courbes time-series")

    foil_options = subset.sort_values("run_score", ascending=False)["foil_id"].astype(str).tolist()
    default_n = min(8, len(foil_options))
    selected_foils = st.multiselect(
        "Foils à afficher",
        options=foil_options,
        default=foil_options[:default_n],
        help="Astuce: sélectionne seulement quelques foils (ex: top 5) pour une lecture plus claire.",
    )

    subset_plot = subset[subset["foil_id"].astype(str).isin([str(x) for x in selected_foils])].copy()
    if subset_plot.empty:
        st.warning("Aucun foil sélectionné (ou pas de données).")
        st.stop()

    # Determine common columns among a few CSVs to propose safe plot choices
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

    preferred = ["Boat.Speed_kts", "Boat.VMG_kts", "Boat.Heel", "Boat.Trim", "Boat.Leeway", "Boat.Helm"]
    candidates = [c for c in preferred if c in common_cols]
    candidates += sorted([c for c in common_cols if c not in candidates])

    y_col = st.selectbox("Colonne à afficher", candidates, index=0)

    # Load series (one run per foil) into long format
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

# # -----------------------------
# # Decision
# # -----------------------------
# with tab_decision:
#     st.subheader("Décision & justification (run-level uniquement)")
#     st.markdown(
#         """
# **Lecture recommandée**
# 1) Regarder le **classement mean_run_score** et les compteurs **good_runs / bad_runs**.  
# 2) Vérifier que le top 2–3 est cohérent sur le **Pareto good_runs vs bad_runs**.  
# 3) Pour un réglage donné : comparer visuellement les time-series (Run detail) :
#    - oscillations (heel/trim/speed)
#    - drawdown
#    - dérive leeway
# """
#     )

#     st.markdown("### Top 5 foils (avec les poids actuels)")
#     cols = ["foil_id", "n_runs", "mean_run_score", "good_runs", "bad_runs", f"top{topk}_mean_run_score", f"top{topk}_std_run_score"]
#     cols = [c for c in cols if c in scored_foils.columns]
#     st.dataframe(scored_foils.head(5)[cols], use_container_width=True, hide_index=True)

#     st.markdown("### Pourquoi le classement change quand on bouge les poids")
#     st.write(
#         "- Si tu augmentes **Performance utile**, tu favorises les runs avec VMG_mean/VMG_95p élevés.\n"
#         "- Si tu augmentes **Stabilité intra-run**, tu favorises les runs “smooth” (variabilité robuste plus faible).\n"
#         "- Si tu augmentes **Dégradation dynamique**, tu pénalises les runs qui s’effondrent ou dérivent."
#     )