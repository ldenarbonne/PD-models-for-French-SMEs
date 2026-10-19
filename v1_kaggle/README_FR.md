# Modèles PD PME France — horizons 1 à 5 ans

## Données
| Rôle | Source | Contenu |
|---|---|---|
| Variables explicatives | Kaggle — *Financial data of French compagnies* (`briaclg/financial-data-of-french-compagnies`, projet Enthic, extraction INPI/RNCS) | 100 000 comptes de résultat, exercices 2012-2016, SIREN, NAF, effectif |
| Défaut | BODACC (DILA, open data) — procédures collectives 2008-2026 | 1er jugement d'ouverture de redressement, liquidation judiciaire ou sauvegarde |
| Date d'observation | BODACC C — dépôts des comptes 2016-2021 | Date réelle de publication des comptes de chaque exercice |

Kaggle ne propose aucun jeu de PME françaises avec indicateur de défaut : la cible est donc reconstruite par appariement SIREN avec le BODACC.

## Choix méthodologiques
- **Date d'observation t0** = publication des comptes au BODACC (au plus tôt clôture + 6 mois). Le jeu Kaggle provient des dépôts INPI à partir de 2017 : avec t0 = clôture + 6 mois, une société n'y figure que si elle a survécu jusqu'à son dépôt, ce qui sous-estime fortement les défauts à 1-3 ans (biais de survie constaté : 1,2 % de défauts cumulés à 2 ans pour la cohorte 2012 contre 4,8 % à 5 ans). Le recalage sur la date de publication corrige ce biais.
- **Défaut** : ouverture de RJ, LJ ou sauvegarde. La Banque de France / l'INSEE retiennent RJ et LJ ; la sauvegarde est ajoutée par prudence (≈ 2 % des ouvertures).
- **Cible h ans** : défaut dans ]t0 ; t0 + h ans]. Sociétés déjà en procédure à t0 exclues.
- **Périmètre PME** : CA ≤ 50 M€, effectif < 250 si renseigné ; exclusion des sections NAF K, O, U.
- **Variables** : 22 candidates issues du compte de résultat (marges, structure des coûts, poids des frais financiers, taille, productivité, délai de publication), secteur et région. Winsorisation 1 %/99 %.
- **Champion** : régression logistique sur WoE (classes par quantiles fusionnées jusqu'à monotonie), présélection IV ≥ 0,02, élimination des corrélations > 0,6, sélection ascendante par BIC, coefficients positifs et p-value < 1 %.
- **Challenger** : LightGBM, nombre d'itérations par validation croisée groupée par SIREN.
- **Validation** : partition 70/30 par SIREN (aucune société des deux côtés) + hors-période (t0 ≥ 01/07/2018).
- **Structure par terme** : PD rendues croissantes avec l'horizon (maximum cumulé), puis échelle de notation à 10 classes sur la PD 1 an.

## Limites
1. Pas de bilan : ni levier, ni liquidité, ni fonds propres — principale cause d'un Gini plus faible que celui des modèles bancaires (souvent 60-70 % à 1 an).
2. Défaut juridique (procédure collective) et non défaut bâlois (90 jours d'impayés, UTP) : PD à recaler avant tout usage IRB/IFRS 9.
3. Calibrage sur le taux moyen de l'échantillon (période 2017-2024, incluant le creux Covid) et non sur une moyenne de long terme couvrant un cycle.
4. Délai de publication : signal pertinent (dépôt tardif = risque) mais partiellement lié à la constitution du jeu Kaggle.
5. Échantillon : sociétés ayant publié leurs comptes ; les sociétés ayant opté pour la confidentialité sont sous-représentées.

## Reproduction
```bash
pip install pandas numpy scikit-learn statsmodels lightgbm matplotlib openpyxl
python 01_build_dataset.py data_kaggle.csv bodacc_open.csv base_pd.csv dossier_dpc/
python 02_models.py base_pd.csv sorties/ 2018-07-01
python 03_report.py sorties/
```
`models.pkl` contient, par horizon, le modèle statsmodels, les variables retenues et l'objet WoE :
`pd_h = model.predict(sm.add_constant(woe.transform(X)[sel], has_constant="add"))`.
