"""
11_build_bilan.py — Base de modélisation PD PME France à partir des données bilantielles.

Sources
- Ratios Financiers (BCE / INPI), DGE, data.economie.gouv.fr, Licence Ouverte 2.0 :
  ratios de bilan et de compte de résultat par SIREN et date de clôture (liasses C et S).
- SIRENE (INSEE, via le miroir Opendatasoft economicref-france-sirene-v3) : date de création,
  forme juridique, tranche d'effectif, NAF, code postal du siège.
- BODACC (DILA) : premier jugement d'ouverture de redressement, liquidation ou sauvegarde.

Conventions
- t0 = date de clôture + 7 mois (approbation sous 6 mois + dépôt sous 1 mois, art. L.232-21 et s.
  du Code de commerce) : on n'utilise que des comptes réputés disponibles à t0.
- Exclusion des sociétés déjà en procédure à t0.
- Cible h ans : défaut dans ]t0 ; t0 + h ans] ; manquante si l'horizon n'est pas entièrement
  observé à la date d'arrêté des données BODACC.
- Comptes « partiellement confidentiels » : le compte de résultat n'est pas publié (montants à 0
  dans la source) ; ces montants sont mis à manquant et la confidentialité devient une variable.
"""
import glob
import sys

import numpy as np
import pandas as pd

RATIOS_DIR, SIRENE, BODACC, OUT = sys.argv[1:5]
DATA_END = pd.Timestamp(sys.argv[5]) if len(sys.argv) > 5 else pd.Timestamp("2026-10-18")

num = ["chiffre_d_affaires", "marge_brute", "ebe", "ebit", "resultat_net", "taux_d_endettement",
       "ratio_de_liquidite", "ratio_de_vetuste", "autonomie_financiere", "poids_bfr_exploitation_sur_ca",
       "couverture_des_interets", "caf_sur_ca", "capacite_de_remboursement", "marge_ebe",
       "resultat_courant_avant_impots_sur_ca", "poids_bfr_exploitation_sur_ca_jours",
       "rotation_des_stocks_jours", "credit_clients_jours", "credit_fournisseurs_jours"]
# ---------------- SIRENE ----------------
s = pd.read_csv(SIRENE, dtype=str, header=None,
                names=["siren", "date_creation", "cat_jur", "naf", "tranche_eff", "cp"])
s = s.drop_duplicates("siren")
s["tranche_eff"] = pd.to_numeric(s["tranche_eff"], errors="coerce").astype("float32")
naf2 = pd.to_numeric(s["naf"].str[:2], errors="coerce")
s["naf_section"] = pd.cut(naf2, [0, 3, 9, 33, 35, 39, 43, 47, 53, 56, 63, 66, 68, 75, 82, 84, 85, 88, 93,
                                 96, 98, 99], labels=list("ABCDEFGHIJKLMNOPQRSTU"), right=True).astype(str)
# périmètre fixe : sociétés commerciales, < 250 salariés, hors K, O, U
s = s[s["cat_jur"].str[:1].eq("5") & ~(s["tranche_eff"] >= 32)
      & ~s["naf_section"].isin(["K", "O", "U", "nan"])]
s = s.rename(columns={"tranche_eff": "eff_tranche"}).drop(columns="naf").set_index("siren")
print("SIRENE dans le périmètre :", len(s))

# ---------------- Ratios, fichier par fichier ----------------
dtypes = {c: "float32" for c in num}
dtypes.update({"siren": str, "type_bilan": "category", "confidentiality": "category"})
parts = []
for f in sorted(glob.glob(RATIOS_DIR + "/r_*.csv")):
    x = pd.read_csv(f, sep=";", dtype=dtypes, usecols=["siren", "date_cloture_exercice", "type_bilan",
                                                        "confidentiality"] + num)
    x = x[x["type_bilan"].isin(["C", "S"])]
    x["siren"] = x["siren"].str.zfill(9)
    x = x[x["siren"].isin(s.index)]
    x["cloture"] = pd.to_datetime(x["date_cloture_exercice"], errors="coerce")
    x = x.drop(columns="date_cloture_exercice")
    x = x[x["cloture"].between("2016-01-01", "2025-12-31")]
    conf = x["confidentiality"].astype(str).str.contains("confidentiel", case=False)
    x.loc[conf, ["chiffre_d_affaires", "marge_brute", "ebe", "ebit"]] = np.nan
    x["conf"] = np.where(conf, "Confidentiel", np.where(x["confidentiality"].astype(str).eq("Public"),
                                                       "Public", "Autre"))
    x = x.drop(columns="confidentiality")
    x = x[~(x["chiffre_d_affaires"] > 50e6) & ~(x["chiffre_d_affaires"] <= 0)]
    parts.append(x)
    print(f, len(x), flush=True)
d = pd.concat(parts, ignore_index=True)
del parts
d = d.drop_duplicates(["siren", "cloture"], keep="last")
d = d.join(s, on="siren", how="inner")
d["type_bilan"] = d["type_bilan"].astype(str)
print("Périmètre PME :", len(d), "| SIREN :", d["siren"].nunique(), flush=True)

# ---------------- Défauts ----------------
b = pd.read_csv(BODACC, dtype=str)
keep = b["nature"].str.contains("liquidation|redressement|sauvegarde|Autre jugement d.ouverture",
                                case=False, regex=True, na=False)
b = b[keep & b["siren"].str.fullmatch(r"\d{9}", na=False)]
b["date"] = pd.to_datetime(b["dateparution"], errors="coerce")
d = d.merge(b.groupby("siren")["date"].min().rename("date_defaut"), left_on="siren",
            right_index=True, how="left")
d["t0"] = d["cloture"] + pd.DateOffset(months=7)
d = d[~(d["date_defaut"] <= d["t0"])]
delta = (d["date_defaut"] - d["t0"]).dt.days / 365.25
for h in range(1, 6):
    y = (delta <= h).astype("float32")
    y[d["t0"] + pd.DateOffset(years=h) > DATA_END] = np.nan
    d[f"def_{h}y"] = y

# ---------------- Variables ----------------
ca = d["chiffre_d_affaires"]
f = pd.DataFrame(index=d.index)
f["autonomie_fin"] = d["autonomie_financiere"]
f["fp_negatifs"] = (d["autonomie_financiere"] < 0).astype(int)
f["endettement"] = d["taux_d_endettement"].where(d["autonomie_financiere"] > 0)
f["liquidite"] = d["ratio_de_liquidite"]
f["vetuste"] = d["ratio_de_vetuste"]
f["bfr_jours"] = d["poids_bfr_exploitation_sur_ca_jours"]
f["couv_interets"] = d["couverture_des_interets"]
f["caf_ca"] = d["caf_sur_ca"]
f["capa_remb"] = d["capacite_de_remboursement"]
f["marge_ebe"] = d["marge_ebe"]
f["rcai_ca"] = d["resultat_courant_avant_impots_sur_ca"]
f["stocks_jours"] = d["rotation_des_stocks_jours"]
f["clients_jours"] = d["credit_clients_jours"]
f["fourn_jours"] = d["credit_fournisseurs_jours"]
f["log_ca"] = np.log(ca)
f["rn_ca"] = d["resultat_net"] / ca
f["ebit_ca"] = d["ebit"] / ca
f["perte"] = (d["resultat_net"] < 0).astype(int)
f["liasse_simplifiee"] = (d["type_bilan"] == "S").astype(int)
f["comptes_confidentiels"] = (d["conf"] == "Confidentiel").astype(int)
d["date_creation"] = pd.to_datetime(d["date_creation"], errors="coerce")
f["age"] = (d["t0"] - d["date_creation"]).dt.days / 365.25
f["log_age"] = np.log1p(f["age"].clip(lower=0))
f.drop(columns="age", inplace=True)
f["eff_tranche"] = d["eff_tranche"]

# évolutions sur un an (exercice précédent du même SIREN)
prev = d[["siren", "cloture", "chiffre_d_affaires", "autonomie_financiere"]].copy()
prev["key"] = prev["cloture"].dt.year + 1
cur = d[["siren"]].assign(key=d["cloture"].dt.year)
m = cur.reset_index().merge(prev.drop(columns="cloture").drop_duplicates(["siren", "key"]),
                            on=["siren", "key"], how="left").set_index("index")
f["croiss_ca"] = d["chiffre_d_affaires"] / m["chiffre_d_affaires"] - 1
f["var_autonomie"] = d["autonomie_financiere"] - m["autonomie_financiere"]

f.replace([np.inf, -np.inf], np.nan, inplace=True)
binary = ["fp_negatifs", "perte", "liasse_simplifiee", "comptes_confidentiels"]
for c in f.columns:
    if c not in binary + ["eff_tranche"]:
        lo, hi = f[c].quantile([0.01, 0.99])
        f[c] = f[c].clip(lo, hi)

cp = d["cp"].fillna("").str[:2]
f["region_grp"] = np.select([cp.isin(["75", "77", "78", "91", "92", "93", "94", "95"]),
                             cp.isin(["13", "06", "83", "84", "04", "05"]),
                             cp.isin(["97"]),
                             cp.isin(["69", "01", "38", "42", "73", "74", "26", "07", "03", "15", "43", "63"])],
                            ["IDF", "PACA", "DOM", "AURA"], default="Autres")
f["forme_jur"] = np.select([d["cat_jur"].str[:2] == "54", d["cat_jur"].str[:2] == "57",
                            d["cat_jur"].str[:2].isin(["55", "56"])], ["SARL", "SAS", "SA"], "Autre")
f["naf_section"] = d["naf_section"]

meta = d[["siren", "cloture", "t0", "date_defaut"] + [f"def_{h}y" for h in range(1, 6)]]
out = pd.concat([meta, f], axis=1)
out.to_csv(OUT, index=False, float_format="%.6g")
print(out.groupby(out["cloture"].dt.year)[[f"def_{h}y" for h in range(1, 6)]]
      .agg(["mean", "count"]).round(4).to_string())
