"""
02_models.py
Estimation de 5 modèles de probabilité de défaut (PD cumulée 1 à 5 ans) pour les PME françaises.

Champion  : régression logistique sur variables discrétisées en WoE (scorecard), une par horizon.
Challenger: gradient boosting (LightGBM), un par horizon, pour mesurer le potentiel non linéaire.

Validation
- Échantillon de développement / test : partition par SIREN (70/30), stratifiée sur le défaut 5 ans,
  pour qu'une même société n'apparaisse jamais des deux côtés.
- Hors-période (OOT) : observations dont la date d'observation est postérieure à OOT_DATE.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
BASE, OUTDIR = sys.argv[1], sys.argv[2]
OOT_DATE = pd.Timestamp(sys.argv[3]) if len(sys.argv) > 3 else pd.Timestamp("2019-01-01")
SEED = 42
H = [1, 2, 3, 4, 5]

df = pd.read_csv(BASE, parse_dates=["t0", "date_defaut"], dtype={"siren": str})

CONT = ["log_ca", "log_eff", "va_ca", "ebe_ca", "rex_ca", "rn_ca", "rcai_ca", "pers_va", "pers_ca",
        "chext_ca", "interets_ca", "chfin_ca", "interets_ebe", "rexcp_ca", "ca_par_eff", "loyer_ca",
        "croiss_ca", "delai_depot"]
BIN = ["eff_missing", "is_impot", "perte", "ebe_neg"]
CAT = ["naf_section", "region_grp"]
CONT = [c for c in CONT if c in df]
FEATS = CONT + BIN + CAT
LABELS = {
    "log_ca": "Taille : log du chiffre d'affaires", "log_eff": "Taille : log(1+effectif)",
    "va_ca": "Taux de valeur ajoutée (VA/CA)", "ebe_ca": "Taux de marge brute d'exploitation (EBE/CA)",
    "rex_ca": "Marge d'exploitation (REX/CA)", "rn_ca": "Marge nette (RN/CA)",
    "rcai_ca": "Résultat courant avant impôt / CA", "pers_va": "Frais de personnel / VA",
    "pers_ca": "Frais de personnel / CA", "chext_ca": "Charges externes / CA",
    "interets_ca": "Intérêts / CA", "chfin_ca": "Charges financières / CA",
    "interets_ebe": "Intérêts / EBE (poids de la dette)", "rexcp_ca": "Résultat exceptionnel / CA",
    "ca_par_eff": "Productivité : log(CA/effectif)", "loyer_ca": "Loyers / CA",
    "croiss_ca": "Croissance du CA sur un an", "delai_depot": "Délai de publication des comptes (mois)",
    "eff_missing": "Effectif non renseigné", "is_impot": "Société imposable (IS > 0)",
    "perte": "Exercice déficitaire", "ebe_neg": "EBE négatif",
    "naf_section": "Secteur (section NAF)", "region_grp": "Région (regroupée)"}

# ------------------------------------------------------------------ partitions
oot_mask = df["t0"] >= OOT_DATE
dev = df[~oot_mask].copy()
oot = df[oot_mask].copy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
tr_idx, te_idx = next(gss.split(dev, groups=dev["siren"]))
train, test = dev.iloc[tr_idx].copy(), dev.iloc[te_idx].copy()
print(f"Train {len(train)} | Test {len(test)} | OOT {len(oot)} (t0 >= {OOT_DATE.date()})")


# ------------------------------------------------------------------ WoE binning
def make_bins(x, y, n=8, min_share=0.05):
    """Découpage par quantiles puis fusion des classes adjacentes jusqu'à monotonie du taux de défaut
    (ou d'une forme en U simple si la monotonie détruit trop d'information)."""
    x = pd.Series(x)
    qs = np.unique(np.nanquantile(x, np.linspace(0, 1, n + 1)))
    if len(qs) <= 2:
        return np.array([-np.inf, np.inf])
    edges = np.concatenate([[-np.inf], qs[1:-1], [np.inf]])
    ok = x.notna()

    def rates(ed):
        b = pd.cut(x[ok], ed, include_lowest=True)
        return y[ok].groupby(b, observed=False).agg(["mean", "size"])

    # fusion des petites classes
    while len(edges) > 3:
        r = rates(edges)
        small = r["size"] < min_share * ok.sum()
        if not small.any():
            break
        i = int(np.argmax(small.values))
        j = i + 1 if i + 1 < len(r) else i      # fusion avec la voisine de droite
        edges = np.delete(edges, j)
    # monotonie
    r = rates(edges)
    direction = np.sign(np.corrcoef(np.arange(len(r)), r["mean"].values)[0, 1]) or 1
    while len(edges) > 3:
        r = rates(edges)["mean"].values
        diffs = np.diff(r) * direction
        if (diffs >= 0).all():
            break
        i = int(np.argmin(diffs))               # pire inversion
        edges = np.delete(edges, i + 1)
    return edges


def woe_table(xb, y, eps=0.5):
    t = pd.crosstab(xb, y).reindex(columns=[0, 1], fill_value=0)
    g, b = t[0] + eps, t[1] + eps
    woe = np.log((b / b.sum()) / (g / g.sum()))      # WoE > 0 = plus risqué
    iv = ((b / b.sum() - g / g.sum()) * woe).sum()
    out = pd.DataFrame({"n": t.sum(1), "defauts": t[1], "taux_defaut": t[1] / t.sum(1), "woe": woe})
    return out, iv


class WoE:
    def fit(self, X, y):
        self.spec, self.tables, self.iv = {}, {}, {}
        for c in FEATS:
            if c in CONT:
                ed = make_bins(X[c], y)
                self.spec[c] = ("cont", ed)
            else:
                self.spec[c] = ("cat", None)
            xb = self._bin(X[c], c)
            tab, iv = woe_table(xb, y)
            self.tables[c], self.iv[c] = tab, iv
        return self

    def _bin(self, x, c):
        kind, ed = self.spec[c]
        if kind == "cont":
            b = pd.cut(x, ed, include_lowest=True).astype(str)
            return b.where(x.notna(), "Manquant")
        return x.astype(str).fillna("Manquant")

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in FEATS:
            xb = self._bin(X[c], c)
            out[c] = xb.map(self.tables[c]["woe"]).fillna(0.0).astype(float)
        return out


# ------------------------------------------------------------------ métriques
def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def hosmer_lemeshow(y, p, g=10):
    q = pd.qcut(p, g, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p}).groupby(q, observed=True).agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
    stat = (((t.o - t.e) ** 2) / (t.e * (1 - t.e / t.n))).sum()
    from scipy.stats import chi2
    return float(stat), float(1 - chi2.cdf(stat, len(t) - 2))


def metrics(y, p):
    hl, hlp = hosmer_lemeshow(y.values, p)
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "defauts": int(y.sum()), "taux_observe": float(y.mean()),
            "pd_moyenne": float(np.mean(p)), "AUC": float(auc), "Gini": float(2 * auc - 1),
            "KS": ks(y, p), "Brier": float(brier_score_loss(y, p)), "HL_stat": hl, "HL_pvalue": hlp}


# ------------------------------------------------------------------ boucle horizons
results, coefs, ivs, preds, deciles, gbm_imp, woe_tabs = [], [], [], {}, [], [], {}
for h in H:
    y = f"def_{h}y"
    trh = train[train[y].notna()]
    ytr = trh[y].astype(int)
    w = WoE().fit(trh[FEATS], ytr)
    woe_tabs[h] = w.tables
    Xtr, Xte, Xoot = w.transform(trh[FEATS]), w.transform(test[FEATS]), w.transform(oot[FEATS])
    iv = pd.Series(w.iv).sort_values(ascending=False)
    ivs.append(iv.rename(f"{h}Y"))

    # présélection : IV >= 0.02, puis élimination des variables corrélées (|rho| > 0.6) par IV décroissant
    cand = [c for c in iv.index if iv[c] >= 0.02]
    keep = []
    corr = Xtr[cand].corr().abs()
    for c in cand:
        if all(corr.loc[c, k] <= 0.6 for k in keep):
            keep.append(c)

    # sélection pas-à-pas ascendante (BIC), signe attendu positif sur la WoE
    sel, best_bic = [], np.inf
    improved = True
    while improved and len(sel) < 12:
        improved, best_c = False, None
        for c in [k for k in keep if k not in sel]:
            m = sm.Logit(ytr, sm.add_constant(Xtr[sel + [c]])).fit(disp=0)
            if m.params[c] <= 0 or m.pvalues[c] > 0.01:
                continue
            if m.bic < best_bic - 2:
                best_bic, best_c = m.bic, c
        if best_c:
            sel.append(best_c)
            improved = True
    # retrait des variables devenues non significatives ou de signe contraire
    while True:
        m = sm.Logit(ytr, sm.add_constant(Xtr[sel])).fit(disp=0)
        bad = [c for c in sel if m.params[c] <= 0 or m.pvalues[c] > 0.01]
        if not bad:
            break
        sel.remove(max(bad, key=lambda c: m.pvalues[c]))
    model = m
    for c in ["const"] + sel:
        coefs.append({"horizon": f"{h}Y", "variable": c, "libelle": LABELS.get(c, "Constante"),
                      "coef": model.params[c], "std_err": model.bse[c], "p_value": model.pvalues[c],
                      "IV": w.iv.get(c, np.nan)})
    vif = {}
    if len(sel) > 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Z = sm.add_constant(Xtr[sel]).values
        vif = {c: variance_inflation_factor(Z, i + 1) for i, c in enumerate(sel)}
    for r in coefs:
        if r["horizon"] == f"{h}Y" and r["variable"] in vif:
            r["VIF"] = vif[r["variable"]]

    p_tr = model.predict(sm.add_constant(Xtr[sel], has_constant="add"))
    p_te = model.predict(sm.add_constant(Xte[sel], has_constant="add"))
    p_oot = model.predict(sm.add_constant(Xoot[sel], has_constant="add"))

    # challenger LightGBM
    Gtr = trh[FEATS].copy()
    for c in CAT:
        Gtr[c] = Gtr[c].astype("category")
    cats = {c: Gtr[c].cat.categories for c in CAT}

    def prep(D):
        D = D[FEATS].copy()
        for c in CAT:
            D[c] = pd.Categorical(D[c], categories=cats[c])
        return D
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  verbose=-1, seed=SEED)
    # nombre d'itérations choisi par validation croisée groupée sur le seul échantillon d'apprentissage
    from sklearn.model_selection import GroupKFold
    dtr = lgb.Dataset(Gtr, ytr, free_raw_data=False)
    folds = list(GroupKFold(n_splits=5).split(Gtr, ytr, groups=trh["siren"]))
    cv = lgb.cv(params, dtr, num_boost_round=2000, folds=folds, metrics="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)])
    n_iter = len(cv["valid auc-mean"])
    gbm = lgb.train(params, dtr, num_boost_round=n_iter)
    g_te, g_oot = gbm.predict(prep(test)), gbm.predict(prep(oot))
    imp = pd.Series(gbm.feature_importance("gain"), index=FEATS)
    gbm_imp.append((imp / imp.sum()).rename(f"{h}Y"))

    mte, moot = test[y].notna().values, oot[y].notna().values
    for name, yy, pp in [("Logit - apprentissage", ytr, p_tr),
                         ("Logit - test", test[y][mte], np.asarray(p_te)[mte]),
                         ("Logit - OOT", oot[y][moot], np.asarray(p_oot)[moot]),
                         ("LightGBM - test", test[y][mte], g_te[mte]),
                         ("LightGBM - OOT", oot[y][moot], g_oot[moot])]:
        if len(yy) and yy.sum() > 0:
            results.append({"horizon": f"{h}Y", "echantillon": name,
                            **metrics(yy.astype(int), np.asarray(pp))})
    preds[h] = {"test": np.asarray(p_te), "oot": np.asarray(p_oot), "train": np.asarray(p_tr),
                "gbm_test": g_te, "model": model, "sel": sel, "woe": w}

    # calibration par décile (test)
    pte_o = np.asarray(p_te)[mte]
    q = pd.qcut(pd.Series(pte_o).rank(method="first"), 10, labels=range(1, 11))
    dd = pd.DataFrame({"decile": q.values, "y": test[y].values[mte], "p": pte_o}) \
        .groupby("decile", observed=True).agg(n=("y", "size"), defauts=("y", "sum"),
                                              taux_observe=("y", "mean"), pd_moyenne=("p", "mean"))
    dd["horizon"] = f"{h}Y"
    deciles.append(dd.reset_index())
    print(f"[{h}Y] variables : {sel}")
    print(pd.DataFrame([r for r in results if r["horizon"] == f"{h}Y"])[
        ["echantillon", "n", "defauts", "taux_observe", "pd_moyenne", "Gini", "KS", "HL_pvalue"]].round(4).to_string())

# ------------------------------------------------------------------ cohérence de la structure par terme
P_te = np.column_stack([preds[h]["test"] for h in H])
viol = (np.diff(P_te, axis=1) < 0).any(axis=1).mean()
P_mono = np.maximum.accumulate(P_te, axis=1)
print(f"Part des sociétés test avec structure par terme non croissante (avant correction) : {viol:.2%}")

# ------------------------------------------------------------------ échelle de notation (sur PD 1 an)
scale = [0, 0.0010, 0.0020, 0.0035, 0.0060, 0.0100, 0.0175, 0.0300, 0.0550, 0.1000, 1.0]
grades = [f"R{i}" for i in range(1, 11)]
allp = pd.concat([test.assign(**{f"pd_{h}y": P_mono[:, i] for i, h in enumerate(H)})])
allp["grade"] = pd.cut(allp["pd_1y"], scale, labels=grades, include_lowest=True)
gt = allp.groupby("grade", observed=False).agg(
    n=("siren", "size"),
    **{f"pd_{h}y_moy": (f"pd_{h}y", "mean") for h in H},
    **{f"dr_{h}y_obs": (f"def_{h}y", "mean") for h in H})
gt["borne_inf_pd1y"], gt["borne_sup_pd1y"] = scale[:-1], scale[1:]

# ------------------------------------------------------------------ export
import pickle
pd.DataFrame(results).to_csv(f"{OUTDIR}/performance.csv", index=False)
pd.DataFrame(coefs).to_csv(f"{OUTDIR}/coefficients.csv", index=False)
pd.concat(ivs, axis=1).rename(index=LABELS).to_csv(f"{OUTDIR}/information_value.csv")
pd.concat(deciles).to_csv(f"{OUTDIR}/calibration_deciles.csv", index=False)
pd.concat(gbm_imp, axis=1).rename(index=LABELS).to_csv(f"{OUTDIR}/lightgbm_importance.csv")
gt.to_csv(f"{OUTDIR}/echelle_notation.csv")
wt = []
for h in H:
    for c in preds[h]["sel"]:
        t = woe_tabs[h][c].reset_index().rename(columns={c: "classe", "row_0": "classe"})
        t.columns = ["classe"] + list(t.columns[1:])
        t.insert(0, "variable", LABELS[c]); t.insert(0, "horizon", f"{h}Y")
        coef = preds[h]["model"].params[c]
        t["coef"] = coef
        t["contribution_logit"] = coef * t["woe"]
        wt.append(t)
pd.concat(wt).to_csv(f"{OUTDIR}/grilles_woe.csv", index=False)
with open(f"{OUTDIR}/models.pkl", "wb") as fh:
    pickle.dump({h: {"model": preds[h]["model"], "sel": preds[h]["sel"], "woe": preds[h]["woe"]}
                 for h in H}, fh)
np.save(f"{OUTDIR}/pred_test.npy", P_mono)
test[["siren", "year", "t0"] + [f"def_{h}y" for h in H]].to_csv(f"{OUTDIR}/test_ids.csv", index=False)
json.dump({"train": len(train), "test": len(test), "oot": len(oot), "oot_date": str(OOT_DATE.date()),
           "term_structure_violation_share": float(viol),
           "default_rates_all": df[[f"def_{h}y" for h in H]].mean().to_dict()},
          open(f"{OUTDIR}/summary.json", "w"), indent=2)
print(gt.round(4).to_string())
