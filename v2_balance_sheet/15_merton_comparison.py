"""
15_merton_comparison.py — Scorecard PD vs Merton PD on French SMEs.

SMEs are unlisted: the market value and volatility of equity required by Merton (1974) are not
observable. The usual "accounting Merton" adaptation is used:
- leverage from the balance sheet: D / V = 1 - equity ratio (total liabilities / total assets);
- asset volatility sigma_V: sector proxy, Damodaran "Std deviation in firm value" (US, Jan. 2026),
  mapped to NACE sections;
- drift mu (physical measure) set to 0 (euro rates close to zero over 2016-2019);
- PD(T) = N(-d2), d2 = [ln(V/D) + (mu - sigma_V^2 / 2) T] / (sigma_V sqrt(T)).

Compared on the same companies (closings 2016-2019, all five horizons observed):
discriminatory power (Gini), level (mean PD vs observed default rate), rank correlation,
an empirical recalibration of the distance to default (KMV-style DD -> EDF mapping) and a
hybrid scorecard + DD model.
"""
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BASE, OUTDIR = sys.argv[1], sys.argv[2]
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
MU = 0.0
H = [1, 2, 3, 4, 5]

SIGMA_V = {  # Damodaran, Std deviation in firm value (US, January 2026)
    "A": 0.374, "B": 0.715, "C": 0.405, "D": 0.172, "E": 0.473, "F": 0.415, "G": 0.325, "H": 0.302,
    "I": 0.342, "J": 0.450, "L": 0.427, "M": 0.358, "N": 0.358, "P": 0.400, "Q": 0.376, "R": 0.434,
    "S": 0.358, "T": 0.358,
}
LABEL2CODE = {
    "Liquidité générale (actif circ. / dettes CT)": "liquidite", "Exercice déficitaire": "perte",
    "Secteur (section NAF)": "naf_section", "Autonomie financière (FP / total bilan)": "autonomie_fin",
    "Charges d'intérêts / EBE": "couv_interets", "Région (regroupée)": "region_grp",
    "Variation de l'autonomie financière sur un an": "var_autonomie", "Forme juridique": "forme_jur",
    "CAF / CA": "caf_ca", "Âge de l'entreprise (log)": "log_age", "Délai fournisseurs (jours)": "fourn_jours",
    "Rotation des stocks (jours)": "stocks_jours", "Croissance du CA sur un an": "croiss_ca",
    "Taille : log du CA": "log_ca"}
INTERVAL = re.compile(r"^[\(\[]\s*(-?inf|[-+0-9.eE]+),\s*(-?inf|[-+0-9.eE]+)\s*\]$")

# ------------------------------------------------------------------ data
it = pd.read_csv(BASE, chunksize=500_000, low_memory=False, parse_dates=["cloture"])
u = lambda s: pd.util.hash_pandas_object(s, index=False).values % 10_000 / 10_000
df = pd.concat([x[(x["cloture"] < "2020-01-01") & (u(x["siren"].astype(str)) < FRAC)] for x in it])
df = df[df["autonomie_fin"].notna() & df["naf_section"].isin(list(SIGMA_V))].reset_index(drop=True)
print("Companies-years :", len(df), "| SIREN :", df["siren"].nunique())

# ------------------------------------------------------------------ scorecard PD (vectorised)
g = pd.read_csv(f"{OUTDIR}/grilles_woe.csv")
c = pd.read_csv(f"{OUTDIR}/coefficients.csv")
g["code"] = g["variable"].map(LABEL2CODE)
g["classe"] = g["classe"].astype(str)
b = g["classe"].str.extract(INTERVAL)
g["lo"] = pd.to_numeric(b[0].str.replace("inf", "Infinity"), errors="coerce")
g["hi"] = pd.to_numeric(b[1].str.replace("inf", "Infinity"), errors="coerce")


def woe_vector(gv, x):
    w = pd.Series(0.0, index=x.index)
    miss = gv[gv["classe"] == "Manquant"]
    if len(miss):
        w[x.isna()] = float(miss["woe"].iloc[0])
    if gv["lo"].notna().any():
        for _, r in gv.dropna(subset=["lo"]).iterrows():
            w[(x > r["lo"]) & (x <= r["hi"])] = r["woe"]
    else:
        key = x.astype(str)
        for _, r in gv[gv["classe"] != "Manquant"].iterrows():
            w[key == r["classe"]] = r["woe"]
    return w


logit_sc = {}
for h in H:
    co = c[c["horizon"] == f"{h}Y"].set_index("variable")["coef"]
    z = pd.Series(float(co["const"]), index=df.index)
    for code, beta in co.drop("const").items():
        z += beta * woe_vector(g[(g["horizon"] == f"{h}Y") & (g["code"] == code)], df[code])
    logit_sc[h] = z
P_sc = np.maximum.accumulate(np.column_stack([1 / (1 + np.exp(-logit_sc[h])) for h in H]), axis=1)

# ------------------------------------------------------------------ Merton PD
a = df["autonomie_fin"].clip(upper=99.9) / 100                # equity / total assets
lev = (1 - a).clip(lower=1e-4)                                # total liabilities / total assets
sv = df["naf_section"].map(SIGMA_V)
ln_vd = -np.log(lev)                                          # ln(V / D)
P_m = np.column_stack([norm.cdf(-(ln_vd + (MU - sv ** 2 / 2) * h) / (sv * np.sqrt(h))) for h in H])
dd1 = (ln_vd + (MU - sv ** 2 / 2)) / sv                       # 1-year distance to default

# ------------------------------------------------------------------ comparison
rows = []
for i, h in enumerate(H):
    y = df[f"def_{h}y"]
    m = y.notna().values
    yy = y[m].astype(int)
    rho = spearmanr(P_sc[m, i], P_m[m, i]).correlation
    rows.append({"Horizon": f"{h}Y", "n": int(m.sum()), "Observed default rate": yy.mean(),
                 "Mean PD scorecard": P_sc[m, i].mean(), "Mean PD Merton": P_m[m, i].mean(),
                 "Median PD Merton": np.median(P_m[m, i]),
                 "Gini scorecard": 2 * roc_auc_score(yy, P_sc[m, i]) - 1,
                 "Gini Merton": 2 * roc_auc_score(yy, P_m[m, i]) - 1,
                 "Spearman rho (scorecard vs Merton)": rho})
comp = pd.DataFrame(rows)

# empirical recalibration of DD (KMV-style mapping) and hybrid model, split by company
gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
tr, te = next(gss.split(df, groups=df["siren"]))
hyb = []
for i, h in enumerate(H):
    y = df[f"def_{h}y"]
    ok = y.notna().values
    trm, tem = np.intersect1d(tr, np.where(ok)[0]), np.intersect1d(te, np.where(ok)[0])
    X_dd = sm.add_constant(pd.DataFrame({"dd": dd1}))
    m_dd = sm.Logit(y.iloc[trm].astype(int), X_dd.iloc[trm]).fit(disp=0)
    p_dd = m_dd.predict(X_dd.iloc[tem])
    X_h = sm.add_constant(pd.DataFrame({"logit_scorecard": logit_sc[h], "dd": dd1}))
    m_h = sm.Logit(y.iloc[trm].astype(int), X_h.iloc[trm]).fit(disp=0)
    p_h = m_h.predict(X_h.iloc[tem])
    yt = y.iloc[tem].astype(int)
    hyb.append({"Horizon": f"{h}Y", "Mean PD Merton recalibrated": p_dd.mean(), "Observed (half-sample)": yt.mean(),
                "Gini scorecard (half-sample)": 2 * roc_auc_score(yt, P_sc[tem, i]) - 1,
                "Gini hybrid scorecard + DD": 2 * roc_auc_score(yt, p_h) - 1,
                "DD coefficient in hybrid": m_h.params["dd"], "DD p-value": m_h.pvalues["dd"]})
hyb = pd.DataFrame(hyb)

# distribution by DD bucket (1 year)
dd_bucket = pd.cut(dd1, [-np.inf, 0, 0.5, 1, 1.5, 2, 3, np.inf])
by_dd = pd.DataFrame({"DD bucket": dd_bucket, "y": df["def_1y"], "pd_m": P_m[:, 0], "pd_sc": P_sc[:, 0]}) \
    .groupby("DD bucket", observed=True).agg(n=("y", "size"), observed=("y", "mean"),
                                             pd_merton=("pd_m", "mean"), pd_scorecard=("pd_sc", "mean"))

comp.to_csv(f"{OUTDIR}/merton_comparison.csv", index=False)
hyb.to_csv(f"{OUTDIR}/merton_hybrid.csv", index=False)
by_dd.to_csv(f"{OUTDIR}/merton_by_dd.csv")
pd.set_option("display.width", 200)
print(comp.round(4).to_string(index=False))
print(hyb.round(4).to_string(index=False))
print(by_dd.round(4).to_string())
