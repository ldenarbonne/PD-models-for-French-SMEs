"""
01_build_dataset.py
Construction de la base de modélisation PD PME France.

Sources
- Kaggle : "Financial data of French compagnies" (briaclg/financial-data-of-french-compagnies),
  projet Enthic, extraction INPI/RNCS : comptes de résultat de sociétés françaises.
- BODACC (DILA, open data) :
    * annonces de procédures collectives (jugements d'ouverture) 2008-2026 -> variable de défaut ;
    * annonces de dépôt des comptes (BODACC C) 2016-2021 -> date réelle de publication des comptes.

Conventions
- Date d'observation t0 = date de publication au BODACC du dépôt des comptes de l'exercice
  (au plus tôt : clôture + 6 mois). Le modèle n'utilise donc que l'information réellement
  disponible à t0, et l'on sait la société vivante à t0.
  Ce choix corrige le biais de survie du jeu Kaggle (comptes extraits des dépôts INPI à partir de 2017 :
  une société n'y figure que si elle a déposé ses comptes après 2017).
- Défaut = première ouverture de redressement judiciaire, liquidation judiciaire ou sauvegarde
  (y compris jugements d'extension) publiée au BODACC.
- Exclusion : sociétés déjà en procédure avant t0.
- Cible horizon h : défaut dans ]t0, t0 + h ans].
"""
import sys
import numpy as np
import pandas as pd

KAGGLE = sys.argv[1]      # data_kaggle.csv
BODACC = sys.argv[2]      # bodacc_open.csv (siren, dateparution, famille, nature)
OUT = sys.argv[3]
DPC = sys.argv[4]         # répertoire contenant dpc_YYYY.csv (siren, date_publication, date_cloture)
DATA_END = pd.Timestamp(sys.argv[5]) if len(sys.argv) > 5 else pd.Timestamp("2026-10-18")

d = pd.read_csv(KAGGLE, low_memory=False)
d = d[d.year.between(2012, 2016)].copy()
d["siren"] = d["siren"].astype(str).str.zfill(9)
d = d.sort_values("Unnamed: 0").drop_duplicates(["siren", "year"], keep="last")

# ---------------- Défauts BODACC ----------------
b = pd.read_csv(BODACC, dtype=str)
keep = b["nature"].str.contains("liquidation|redressement|sauvegarde|Autre jugement d.ouverture",
                                case=False, regex=True, na=False)
b = b[keep & b["siren"].str.fullmatch(r"\d{9}", na=False)]
b["date"] = pd.to_datetime(b["dateparution"], errors="coerce")
first_def = b.groupby("siren")["date"].min().rename("date_defaut")
d = d.merge(first_def, left_on="siren", right_index=True, how="left")

# ---------------- Date de publication des comptes (BODACC C) ----------------
import glob
dp = pd.concat([pd.read_csv(f, header=None, names=["siren", "pub", "cloture"], dtype=str)
                for f in glob.glob(DPC + "/dpc_20*.csv")])
dp["pub"] = pd.to_datetime(dp["pub"], errors="coerce")
dp["cloture"] = pd.to_datetime(dp["cloture"], errors="coerce")
dp = dp.dropna()
dp["year"] = dp["cloture"].dt.year
dp = dp.sort_values("pub").drop_duplicates(["siren", "year"], keep="first")
n0 = len(d)
d["year"] = d["year"].astype(int)
d = d.merge(dp[["siren", "year", "pub", "cloture"]], on=["siren", "year"], how="inner")
print(f"Appariement dépôt des comptes : {len(d)}/{n0} lignes ({len(d)/n0:.1%})")
d["t0"] = np.maximum(d["pub"], d["cloture"] + pd.DateOffset(months=6))
d["delai_depot"] = (d["pub"] - d["cloture"]).dt.days / 30.44

d = d[~(d["date_defaut"] <= d["t0"])]                       # déjà en procédure à t0
delta = (d["date_defaut"] - d["t0"]).dt.days / 365.25
for h in range(1, 6):
    d[f"def_{h}y"] = (delta <= h).astype(int)
    # horizon non entièrement observé -> cible manquante
    d.loc[d["t0"] + pd.DateOffset(years=h) > DATA_END, f"def_{h}y"] = np.nan

# ---------------- Périmètre PME ----------------
CA = "Chiffres d’affaires nets"
EFF = "Effectif moyen du personnel"
d = d[d[CA] > 0]
d = d[d[CA] <= 50e6]                                         # seuil CA PME (Reco. UE 2003/361)
d = d[~(d[EFF] >= 250)]                                      # effectif < 250 quand renseigné

ape = pd.read_csv(KAGGLE.replace("data_kaggle.csv", "ape_fusion.csv"))
div2sec = (ape.assign(div=ape["ape_cat0"].astype(int))
              .drop_duplicates("div").set_index("div")["Code"])
d["naf_div"] = pd.to_numeric(d["ape_division"], errors="coerce")
d["naf_section"] = d["naf_div"].map(div2sec).fillna("NA")
# Exclusions sectorielles usuelles : finance/assurance (K), administration publique (O),
# organisations extraterritoriales (U)
d = d[~d["naf_section"].isin(["K", "O", "U"])]

d["dep"] = d["postal_code"].astype(str).str.extract(r"^(\d{2})")[0]
d["region_grp"] = np.select(
    [d["dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"]),
     d["dep"].isin(["13", "06", "83", "84", "04", "05"]),
     d["dep"].isin(["97"])],
    ["IDF", "PACA", "DOM"], default="Autres")

# ---------------- Ratios (compte de résultat) ----------------
def g(c):
    return d[c].fillna(0) if c in d else 0

ca = d[CA]
prod = ca + g("Production stockée") + g("Production immobilisée")
conso = (g("Achats de marchandises (y compris droits de douane)") + g("Variation de stock (marchandises)")
         + g("Achats de matières premières et autres approvisionnements")
         + g("Variation de stock (matières premières et approvisionnements)")
         + g("Autres achats et charges externes"))
va = prod - conso
pers = g("Salaires et traitements") + g("Charges sociales")
ebe = va + g("Subventions d’exploitation") - g("Impôts, taxes et versements assimilés") - pers
rex = d["Résultat d'exploitation"]
rex = rex.fillna(g("Total des produits d’exploitation") - g("Total des charges d’exploitation"))
rn = d["Bénéfices ou perte (Total des produits ‐ Total des charges)"]
interets = g("Intérêts et charges assimilées")
chfin = g("Total des charges financières")

f = pd.DataFrame(index=d.index)
f["log_ca"] = np.log(ca)
f["log_eff"] = np.log1p(d[EFF])
f["eff_missing"] = d[EFF].isna().astype(int)
f["va_ca"] = va / ca
f["ebe_ca"] = ebe / ca
f["rex_ca"] = rex / ca
f["rn_ca"] = rn / ca
f["rcai_ca"] = d["Résultat en cours avant impôts"] / ca
f["pers_va"] = np.where(va > 0, pers / va, np.nan)
f["pers_ca"] = pers / ca
f["chext_ca"] = g("Autres achats et charges externes") / ca
f["interets_ca"] = interets / ca
f["chfin_ca"] = chfin / ca
f["interets_ebe"] = np.where(ebe > 0, interets / ebe, np.nan)   # poids de la dette (proxy)
f["rexcp_ca"] = d["Résultat exceptionnel"] / ca
f["is_impot"] = (g("Impôts sur les bénéfices") > 0).astype(int)
f["perte"] = (rn < 0).astype(int)
f["ebe_neg"] = (ebe < 0).astype(int)
f["ca_par_eff"] = np.log(ca / d[EFF].where(d[EFF] > 0))
f["delai_depot"] = d["delai_depot"]
f["loyer_ca"] = g("Location, charges locatives et de copropriété") / ca

# Croissance du CA quand l'exercice précédent est disponible
d_sorted = d[["siren", "year"]].assign(ca=ca)
prev = d_sorted.rename(columns={"ca": "ca_prev"}).assign(year=lambda x: x.year + 1)
d_sorted = d_sorted.merge(prev, on=["siren", "year"], how="left")
f["croiss_ca"] = (d_sorted["ca"].values / d_sorted["ca_prev"].values) - 1
f.replace([np.inf, -np.inf], np.nan, inplace=True)

# Winsorisation 1 % / 99 % des ratios continus
cont = [c for c in f.columns if c not in ("eff_missing", "is_impot", "perte", "ebe_neg")]
for c in cont:
    lo, hi = f[c].quantile([0.01, 0.99])
    f[c] = f[c].clip(lo, hi)

meta = d[["siren", "denomination", "year", "cloture", "pub", "t0", "date_defaut", "naf_section", "region_grp",
          "dep", "is_ess"] + [f"def_{h}y" for h in range(1, 6)]]
out = pd.concat([meta, f], axis=1)
out.to_csv(OUT, index=False)

print("Observations :", len(out), "| SIREN :", out.siren.nunique())
print(out.groupby("year").size())
print("Taux de défaut cumulés :")
print(out[[f"def_{h}y" for h in range(1, 6)]].mean().round(4))
print(out.groupby("year")[[f"def_{h}y" for h in range(1, 6)]].mean().round(4))
