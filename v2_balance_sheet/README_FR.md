# Modèles PD PME France — version 2 (données bilantielles)

## Données
| Rôle | Source | Contenu |
|---|---|---|
| Variables financières | « Ratios Financiers (BCE / INPI) », DGE, data.economie.gouv.fr — Licence Ouverte 2.0 | Ratios de bilan et de compte de résultat par SIREN et date de clôture, liasses complètes (C) et simplifiées (S), clôtures 2016-2025 |
| Identification | SIRENE (INSEE), miroir Opendatasoft `economicref-france-sirene-v3` | Date de création, forme juridique, NAF, code postal du siège, tranche d'effectif |
| Défaut | BODACC (DILA), annonces de procédures collectives | Premier jugement d'ouverture de redressement, liquidation judiciaire ou sauvegarde |

## Choix méthodologiques
- **t0 = clôture + 7 mois** : délai légal d'approbation (6 mois) et de dépôt (1 mois) des comptes.
- **Défaut** : ouverture de RJ, LJ ou sauvegarde publiée au BODACC ; sociétés déjà en procédure à t0 exclues.
- **Cible h ans** : défaut dans ]t0 ; t0 + h ans], manquante si l'horizon n'est pas entièrement observé (BODACC arrêté au 18/10/2026).
- **Périmètre PME** : sociétés commerciales (catégorie juridique 5xxx), CA ≤ 50 M€, moins de 250 salariés, CA > 0 pour les comptes publics, hors NAF K, O, U.
- **Comptes confidentiels** : compte de résultat non publié → montants mis à manquant ; la confidentialité devient une variable explicative.
- **Variables candidates** : autonomie financière, fonds propres négatifs, endettement, liquidité, vétusté, BFR, délais clients et fournisseurs, rotation des stocks, couverture des intérêts, CAF/CA, capacité de remboursement, marges, taille, âge, croissance du CA, variation de l'autonomie financière, type de liasse, confidentialité, secteur, région, forme juridique. Winsorisation 1 %/99 %.
- **Tranche d'effectif SIRENE** : utilisée uniquement pour le périmètre (information actuelle, non datée : risque de fuite d'information si elle servait de variable).
- **Champion** : régression logistique sur WoE ; sélection ascendante par BIC sur un sous-échantillon de 400 000 lignes, estimation finale sur l'échantillon d'apprentissage complet.
- **Challenger** : LightGBM.
- **Validation** : développement = clôtures 2016-2019, partition 70/30 par SIREN ; hors-période = clôtures 2020 et après.

## Reproduction
```bash
python 11_build_bilan.py dossier_ratios/ sirene.csv bodacc_open.csv base_bilan.csv
python 12_models_bilan.py base_bilan.csv sorties/ 2020-01-01
python 13_report_bilan.py sorties/ ../sorties/performance.csv
```
