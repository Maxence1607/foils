import json
from collections import defaultdict

# Charger le fichier
with open("data/configs.json", "r") as f:
    data = json.load(f)

# 1️⃣ Dictionnaire foil -> liste des scripts
foil_to_scripts = defaultdict(list)

for script_name, content in data.items():
    foil = content["Configuration"]["Foil"]
    foil_to_scripts[foil].append(script_name)

# Convertir en dict classique (optionnel)
foil_to_scripts = dict(foil_to_scripts)

# 2️⃣ Dictionnaire foil -> nombre de scripts
foil_counts = {foil: len(scripts) for foil, scripts in foil_to_scripts.items()}

print("Foil -> scripts :")
print(foil_to_scripts)

print("\nFoil -> nombre de scripts :")
print(foil_counts)





import os
import glob
import json
import pandas as pd
from collections import defaultdict

SIM_DIR = "data/simulations"
JSON_FILE = "data/configs.json"  # ton fichier json script -> foil

# -----------------------------
# 1️⃣ Charger le mapping script -> foil
# -----------------------------

with open(JSON_FILE, "r") as f:
    mapping = json.load(f)

# mapping = foil_to_scripts

script_to_foil = {
    script: content["Configuration"]["Foil"]
    for script, content in mapping.items()
}

print(script_to_foil)
# -----------------------------
# 2️⃣ Fonction pour extraire paramètres constants d’un CSV
# -----------------------------

def get_constant_columns(df):
    constants = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 0 and s.nunique() == 1:
            constants[col] = s.iloc[0]
    return constants

# -----------------------------
# 3️⃣ Construire un dict foil -> liste de paramètres par simulation
# -----------------------------

foil_records = defaultdict(list)

for file in glob.glob(os.path.join(SIM_DIR, "*.csv")):
    
    script_name = os.path.basename(file)
    # print(script_name)
    
    # print(script_to_foil)
    script_name = script_name.replace('modified_', '').replace('.csv', '.js')
    if script_name not in script_to_foil:
        continue  # skip si pas dans mapping
    
    foil = script_to_foil[script_name]
    
    df = pd.read_csv(file)
    
    constants = get_constant_columns(df)
    
    record = {"script": script_name, **constants}
    
    foil_records[foil].append(record)

# -----------------------------
# 4️⃣ Analyse par foil
# -----------------------------

for foil, records in foil_records.items():
    
    print(f"\n====================")
    print(f"FOIL {foil}")
    print(f"====================")
    
    df_params = pd.DataFrame(records).set_index("script")
    
    nunique = df_params.nunique().sort_values(ascending=False)
    
    print("\nParamètres qui changent :")
    print(nunique[nunique > 1])
    
    print("\nParamètres constants pour ce foil :")
    print(nunique[nunique == 1])








# -----------------------------
# Paramètres qui changent :
# Boat.Aero.Travel      5
# Boat.Keel.KeelCant    4
# Boat.Port.FoilRake    3

# Donc probablement l'origine des 60 simulations par foil (5*4*3), ils ont fait 
# toutes les combinaisons possibles de ces 3 variables pour générer l'ensemble des simulations.
# Le code qui suit sert à vérifier cette hypothèse
# -----------------------------



import os
import glob
import json
import pandas as pd
from itertools import product

SIM_DIR = "data/simulations"
JSON_FILE = "data/configs.json"   # adapte le nom
FOIL_TARGET = "02"           # <-- mets ici le foil que tu veux tester

COLS = ["Boat.Aero.Travel", "Boat.Keel.KeelCant", "Boat.Port.FoilRake"]

# --- 1) mapping script -> foil
with open(JSON_FILE, "r") as f:
    mapping = json.load(f)

script_to_foil = {k: v["Configuration"]["Foil"] for k, v in mapping.items()}

def constant_value_in_file(csv_path: str, col: str):
    """Retourne la valeur unique d'une colonne dans un fichier (en ignorant NaN).
       Renvoie None si colonne absente ou non-constante."""
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    vals = s.unique()
    if len(vals) == 1:
        return vals[0]
    return None

# --- 2) construire la table des paramètres pour ce foil
rows = []
for path in sorted(glob.glob(os.path.join(SIM_DIR, "*.csv"))):
    script = os.path.basename(path)
    print(script)
    script = script.replace('modified_', '').replace('.csv', '.js')
    if script_to_foil.get(script) != FOIL_TARGET:
        continue

    rec = {"script": script}
    for c in COLS:
        rec[c] = constant_value_in_file(path, c)
    rows.append(rec)

print(pd.DataFrame(rows))
df_params = pd.DataFrame(rows).set_index("script").sort_index()

print("Nombre de simulations trouvées pour ce foil :", len(df_params))
print("\nAperçu :")
print(df_params.head())

# --- 3) valeurs uniques par colonne (sur l'ensemble des 60 sims)
unique_vals = {c: sorted(df_params[c].dropna().unique().tolist()) for c in COLS}
print("\nValeurs uniques par colonne :")
for c, vals in unique_vals.items():
    print(f"- {c} ({len(vals)}): {vals}")

expected = len(unique_vals[COLS[0]]) * len(unique_vals[COLS[1]]) * len(unique_vals[COLS[2]])
observed = df_params.dropna(subset=COLS).drop_duplicates(subset=COLS).shape[0]

print(f"\nCombinaisons attendues (produit) : {expected}")
print(f"Combinaisons observées (distinctes) : {observed}")

# --- 4) afficher, pour chaque simulation, les 3 valeurs (ce que tu demandes)
print("\nValeurs par simulation (script -> 3 paramètres) :")
print(df_params[COLS].sort_values(COLS).to_string())

# --- 5) vérifier couverture complète: combinaisons manquantes / doublons
# (a) combinaisons manquantes
all_expected = pd.DataFrame(
    list(product(unique_vals[COLS[0]], unique_vals[COLS[1]], unique_vals[COLS[2]])),
    columns=COLS
)

observed_df = df_params.dropna(subset=COLS).reset_index()[["script"] + COLS]
observed_unique = observed_df.drop_duplicates(subset=COLS)

missing = all_expected.merge(observed_unique[COLS], on=COLS, how="left", indicator=True)
missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])

print(f"\nNombre de combinaisons manquantes : {len(missing)}")
if len(missing) > 0:
    print(missing.to_string(index=False))

# (b) doublons: plusieurs scripts pour la même combinaison
dup = observed_df.groupby(COLS).size().reset_index(name="n_scripts")
dup = dup[dup["n_scripts"] > 1].sort_values("n_scripts", ascending=False)

print(f"\nNombre de combinaisons dupliquées : {len(dup)}")
if len(dup) > 0:
    print(dup.to_string(index=False))
    # Optionnel: lister les scripts impliqués pour une combinaison dupliquée
    # print(observed_df.merge(dup[COLS], on=COLS, how="inner").sort_values(COLS).to_string(index=False))




# -----------------------------

# Donc probablement l'origine des 60 simulations par foil (5*4*3), ils ont fait L'hypothèse précédente est validée, les simulations représentent bien l'ensemble des combinaisons des 3 colonnes mentionnées.
# Maintenant on va vérifier que les simulations de tous les foils ont été faites pour TWS/TWA constants sur les 60 fichiers pour chaque foil
# Le code qui suit sert à vérifier cette hypothèse
# -----------------------------




import os, glob, json
import pandas as pd

SIM_DIR = "data/simulations"
JSON_FILE = "data/configs.json"
FOIL_TARGET = "02"  # adapte

# Colonnes de "condition" typiques (mets-en d'autres si tu en as)
COND_COLS = ["Boat.TWS_kts", "Boat.TWA"]  # + éventuellement "Boat.TWS_kts", "Boat.TWA", "SeaState", etc.

with open(JSON_FILE, "r") as f:
    mapping = json.load(f)

script_to_foil = {k: v["Configuration"]["Foil"] for k, v in mapping.items()}

def constant_value(csv_path, col):
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    vals = s.unique()
    return vals[0] if len(vals) == 1 else "NOT_CONSTANT"

rows = []
for path in sorted(glob.glob(os.path.join(SIM_DIR, "*.csv"))):
    script = os.path.basename(path)
    script = script.replace('modified_', '').replace('.csv', '.js')
    if script_to_foil.get(script) != FOIL_TARGET:
        continue
    rec = {"script": script}
    for c in COND_COLS:
        rec[c] = constant_value(path, c)
    rows.append(rec)

df_cond = pd.DataFrame(rows).set_index("script").sort_index()

print(df_cond.head(10))
print("\nNombre de sims:", len(df_cond))

for c in COND_COLS:
    vals = df_cond[c].dropna().unique()
    print(f"\n{c} - nb valeurs distinctes: {len(vals)}")
    print(vals[:20])


print(df["Boat.TWA"].describe())
for i, row in df.iterrows():
    print(row['Boat.TWA'])