"""
Streamlit app — PD calculator for French SMEs (1- to 5-year horizons).

The app applies the five WoE logistic scorecards estimated in v2_balance_sheet/
(files grilles_woe.csv and coefficients.csv). No raw data is needed.
"""
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "v2_balance_sheet" / "outputs"
H = [1, 2, 3, 4, 5]

# Label used in grilles_woe.csv -> internal code
LABEL2CODE = {
    "Liquidité générale (actif circ. / dettes CT)": "liquidite",
    "Exercice déficitaire": "perte",
    "Secteur (section NAF)": "naf_section",
    "Autonomie financière (FP / total bilan)": "autonomie_fin",
    "Charges d'intérêts / EBE": "couv_interets",
    "Région (regroupée)": "region_grp",
    "Variation de l'autonomie financière sur un an": "var_autonomie",
    "Forme juridique": "forme_jur",
    "CAF / CA": "caf_ca",
    "Âge de l'entreprise (log)": "log_age",
    "Délai fournisseurs (jours)": "fourn_jours",
    "Rotation des stocks (jours)": "stocks_jours",
    "Croissance du CA sur un an": "croiss_ca",
    "Taille : log du CA": "log_ca",
}
DISPLAY = {
    "liquidite": "Current ratio", "perte": "Loss-making year", "naf_section": "Sector",
    "autonomie_fin": "Equity ratio", "couv_interets": "Interest / EBITDA", "region_grp": "Region",
    "var_autonomie": "Change in equity ratio", "forme_jur": "Legal form", "caf_ca": "Cash flow / sales",
    "log_age": "Company age", "fourn_jours": "Payables days", "stocks_jours": "Inventory days",
    "croiss_ca": "Sales growth", "log_ca": "Size (turnover)",
}
SECTORS = {
    "A": "A – Agriculture, forestry, fishing", "B": "B – Mining and quarrying", "C": "C – Manufacturing",
    "D": "D – Electricity, gas", "E": "E – Water, waste management", "F": "F – Construction",
    "G": "G – Wholesale and retail trade", "H": "H – Transport and storage",
    "I": "I – Accommodation and food services", "J": "J – Information and communication",
    "L": "L – Real estate", "M": "M – Professional, scientific and technical",
    "N": "N – Administrative and support services", "P": "P – Education",
    "Q": "Q – Human health and social work", "R": "R – Arts, entertainment, recreation",
    "S": "S – Other services", "T": "T – Households as employers",
}
REGIONS = {"IDF": "Île-de-France", "AURA": "Auvergne-Rhône-Alpes", "PACA": "Provence-Alpes-Côte d'Azur",
           "DOM": "Overseas departments", "Autres": "Other regions"}
FORMS = {"SAS": "SAS / SASU", "SARL": "SARL / EURL", "SA": "SA", "Autre": "Other commercial company"}
SCALE = [0, 0.0010, 0.0020, 0.0035, 0.0060, 0.0100, 0.0175, 0.0300, 0.0550, 0.1000, 1.0]
GRADES = [f"R{i}" for i in range(1, 11)]
INTERVAL = re.compile(r"^[\(\[]\s*(-?inf|[-+0-9.eE]+),\s*(-?inf|[-+0-9.eE]+)\s*\]$")


@st.cache_data
def load_scorecards():
    g = pd.read_csv(OUT / "grilles_woe.csv")
    c = pd.read_csv(OUT / "coefficients.csv")
    g["code"] = g["variable"].map(LABEL2CODE)
    g["classe"] = g["classe"].astype(str)
    bounds = g["classe"].str.extract(INTERVAL)
    g["lo"] = pd.to_numeric(bounds[0].str.replace("inf", "Infinity"), errors="coerce")
    g["hi"] = pd.to_numeric(bounds[1].str.replace("inf", "Infinity"), errors="coerce")
    return g, c


def woe_of(g_hv, value):
    """WoE of a value for one variable and one horizon (same rules as the training code)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        row = g_hv[g_hv["classe"] == "Manquant"]
        return (float(row["woe"].iloc[0]), "Missing") if len(row) else (0.0, "Missing")
    numeric = g_hv["lo"].notna().any()
    if numeric:
        hit = g_hv[(g_hv["lo"] < value) & (value <= g_hv["hi"])]
        if len(hit):
            return float(hit["woe"].iloc[0]), hit["classe"].iloc[0]
        return 0.0, "out of range"
    key = str(value)
    hit = g_hv[g_hv["classe"] == key]
    return (float(hit["woe"].iloc[0]), key) if len(hit) else (0.0, key)


def score(inputs, g, c):
    rows, pds = [], []
    for h in H:
        hz = f"{h}Y"
        coefs = c[c["horizon"] == hz].set_index("variable")["coef"]
        logit = float(coefs["const"])
        for code, beta in coefs.drop("const").items():
            w, cls = woe_of(g[(g["horizon"] == hz) & (g["code"] == code)], inputs.get(code))
            logit += beta * w
            rows.append({"Horizon": hz, "Variable": DISPLAY[code], "Bin": cls,
                         "Contribution to logit": beta * w})
        pds.append(1 / (1 + np.exp(-logit)))
    pds = np.maximum.accumulate(np.array(pds))          # non-decreasing term structure
    return pds, pd.DataFrame(rows)


def grade(pd1):
    return GRADES[int(np.searchsorted(SCALE, pd1, side="left")) - 1] if pd1 > 0 else "R1"


# ------------------------------------------------------------------ page
st.set_page_config(page_title="PD models — French SMEs", page_icon="📉", layout="wide")
st.title("Probability of default — French SMEs")
st.caption("Five WoE logistic scorecards (1- to 5-year cumulative PD) estimated on French open data "
           "(BCE/INPI financial ratios, SIRENE, BODACC). Illustrative research model, not a credit decision tool.")

g, c = load_scorecards()
tab_calc, tab_perf, tab_about = st.tabs(["PD calculator", "Model performance", "Methodology"])

with tab_calc:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Company profile")
        sector = st.selectbox("Sector (NACE section)", list(SECTORS), format_func=SECTORS.get, index=6)
        form = st.selectbox("Legal form", list(FORMS), format_func=FORMS.get)
        region = st.selectbox("Region", list(REGIONS), format_func=REGIONS.get, index=4)
        age = st.number_input("Company age (years)", 0.0, 150.0, 8.0, 0.5)
        turnover = st.number_input("Turnover (€)", 1_000.0, 50_000_000.0, 1_500_000.0, 50_000.0, format="%.0f")

        st.subheader("Financial statements")
        equity = st.number_input("Equity ratio — equity / total assets (%)", -200.0, 100.0, 30.0, 1.0,
                                 help="Negative if equity is negative.")
        liquidity = st.number_input("Current ratio — current assets / short-term debt (%)", 0.0, 2000.0, 150.0, 5.0)
        caf = st.number_input("Cash flow (CAF) / sales (%)", -100.0, 100.0, 6.0, 0.5)
        interest = st.number_input("Interest expense / EBITDA (%)", -500.0, 500.0, 5.0, 1.0,
                                   help="Negative when EBITDA is negative. Enter 0 if no financial debt.")
        payables = st.number_input("Payables days", 0.0, 720.0, 60.0, 5.0)
        inventory = st.number_input("Inventory days (0 if no stock)", 0.0, 720.0, 20.0, 5.0)
        loss = st.toggle("Net loss in the last fiscal year", value=False)

        st.subheader("Year-on-year change")
        has_prev = st.toggle("Previous fiscal year available", value=True)
        growth = st.number_input("Sales growth (%)", -100.0, 500.0, 3.0, 1.0, disabled=not has_prev)
        d_equity = st.number_input("Change in equity ratio (points)", -100.0, 100.0, 1.0, 0.5, disabled=not has_prev)

    inputs = {
        "naf_section": sector, "forme_jur": form, "region_grp": region,
        "log_age": float(np.log1p(age)), "log_ca": float(np.log(turnover)),
        "autonomie_fin": equity, "liquidite": liquidity, "caf_ca": caf, "couv_interets": interest,
        "fourn_jours": payables, "stocks_jours": inventory, "perte": 1.0 if loss else 0.0,
        "croiss_ca": growth / 100 if has_prev else None, "var_autonomie": d_equity if has_prev else None,
    }
    pds, contrib = score(inputs, g, c)

    with right:
        st.subheader("Result")
        k = st.columns(3)
        k[0].metric("PD 1 year", f"{pds[0]:.2%}")
        k[1].metric("PD 5 years", f"{pds[4]:.2%}")
        k[2].metric("Rating grade (1-year PD)", grade(pds[0]))

        ts = pd.DataFrame({"Horizon (years)": H, "Cumulative PD (%)": pds * 100})
        line = alt.Chart(ts).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2, color="#2a78d6").encode(
            x=alt.X("Horizon (years):O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Cumulative PD (%):Q", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("Horizon (years):O"), alt.Tooltip("Cumulative PD (%):Q", format=".2f")],
        ).properties(height=280, title="PD term structure")
        st.altair_chart(line, width="stretch")

        marg = np.diff(np.concatenate([[0], pds]))
        st.dataframe(pd.DataFrame({"Horizon": [f"{h}Y" for h in H],
                                   "Cumulative PD": [f"{p:.2%}" for p in pds],
                                   "Marginal PD": [f"{m:.2%}" for m in marg]}),
                     hide_index=True, width="stretch")

        st.subheader("Score breakdown")
        hz = st.radio("Horizon", [f"{h}Y" for h in H], horizontal=True)
        cb = contrib[contrib["Horizon"] == hz].sort_values("Contribution to logit")
        bars = alt.Chart(cb).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Contribution to logit:Q", title="Contribution to logit (> 0 increases risk)"),
            y=alt.Y("Variable:N", sort=list(cb["Variable"]), title=None),
            color=alt.condition("datum['Contribution to logit'] > 0", alt.value("#eb6834"), alt.value("#2a78d6")),
            tooltip=["Variable", "Bin", alt.Tooltip("Contribution to logit:Q", format="+.3f")],
        ).properties(height=34 * len(cb))
        st.altair_chart(bars, width="stretch")
        st.caption("Orange bars push the PD up, blue bars push it down, relative to an average company.")

with tab_perf:
    perf = pd.read_csv(OUT / "performance.csv")
    comp = pd.read_csv(OUT / "comparaison_v1_v2.csv")
    st.subheader("Discriminatory power (Gini)")
    show = comp.rename(columns={"Horizon": "Horizon"}).copy()
    for col in show.columns[1:4]:
        show[col] = show[col].map("{:.1%}".format)
    show["Gain (points)"] = show["Gain (points)"].map("{:+.1f}".format)
    show.columns = ["Horizon", "V1 — P&L only (Kaggle)", "V2 — balance sheet (test)",
                    "V2 — balance sheet (out-of-time)", "Gain (points)"]
    st.dataframe(show, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(OUT / "fig_roc.png"), caption="ROC curves — test sample")
    c2.image(str(OUT / "fig_term_structure.png"), caption="Average cumulative PD by rating grade")
    st.image(str(OUT / "fig_calibration.png"), caption="Calibration by risk decile — test sample")
    with st.expander("Full performance table"):
        st.dataframe(perf.round(4), hide_index=True, width="stretch")

with tab_about:
    st.markdown("""
**Data.** Financial ratios from *Ratios Financiers (BCE / INPI)* (French Ministry of Economy, Licence Ouverte),
company register SIRENE (INSEE), court insolvency openings from BODACC (DILA). Closings 2016–2025,
about 5 million company-years after cleaning.

**Default.** First court opening of safeguard, receivership or liquidation (legal default, not Basel 90 days past due).

**Observation date.** Fiscal year-end + 7 months (legal filing deadline).

**Scope.** Commercial companies, turnover ≤ €50M, fewer than 250 employees, excluding finance, public administration
and extraterritorial bodies.

**Model.** One logistic regression on Weight-of-Evidence variables per horizon (monotonic binning, IV ≥ 0.02,
correlation filter, forward selection on BIC with expected-sign constraints). PDs are made non-decreasing across horizons.

**Validation.** Development on closings 2016–2019 (70/30 split by company), out-of-time test on closings from 2020.

**Main caveat.** On post-2020 data the models rank firms well but underestimate the default level (state-guaranteed
loans, then the 2023–2024 insolvency surge): a central-tendency recalibration would be required before any real use.

Code and documentation: [GitHub repository](https://github.com/ldenarbonne/PD-models-for-French-SMEs) ·
Author: [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/)
""")
