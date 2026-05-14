# Battery_Diagnostics_Thesis
 Data-driven battery diagnostics using EIS

**Author:** Bala Manikanta Yaswanth Kanagarla
**Institution:** Technische Hochschule Ingolstadt (THI)
**Supervisor:** Prof. Dr. H.-G. Schweiger
**Period:** 30th October 2025 – 27th April 2026

---

## Overview

This project implements a machine learning framework for 
non-invasive State of Health (SoH) estimation of lithium-ion 
batteries using Electrochemical Impedance Spectroscopy (EIS).

Eight physically interpretable features are extracted from 
impedance spectra and used to train six ML models under a 
strict cell-level train-test split. The stacking ensemble 
achieves R² = 0.9122 and RMSE = 0.0234 on completely unseen 
cells.

---

## Repository Structure
battery-soh-estimation/
│
├── data/
│   ├── raw/                  # Original EIS and Capacity folders
│   │   ├── Capacity/         # Raw capacity measurement files
│   │   ├── Capacity_1/       # Additional capacity files
│   │   ├── EIS/              # Full EIS dataset
│   │   ├── EIS_state_II/     # EIS measurements at State II
│   │   ├── EIS_state_III/    # EIS measurements at State III
│   │   ├── EIS_state_IV/     # EIS measurements at State IV
│   │   ├── EIS_state_V/      # EIS measurements at State V
│   │   ├── EIS_state_VI/     # EIS measurements at State VI
│   │   ├── EIS_state_VII/    # EIS measurements at State VII
│   │   ├── EIS_state_VIII/   # EIS measurements at State VIII
│   │   ├── EIS_state_IX/     # EIS measurements at State IX
│   │   └── EIS_state_V_IX/   # Combined State V and IX
│   │
│   ├── processed/            # Cleaned and merged CSV files
│   │   ├── merged_eis_capacity.csv
│   │   ├── merged_eis_capacity_final.csv
│   │   ├── merged_eis_capacity_final_norm.csv
│   │   ├── merged_eis_capacity_stateV.csv
│   │   └── merged_eis_capacity_stateV_IX.csv
│   │
│   └── features/             # Final feature datasets per state
│       ├── eis_features_all_cells.csv
│       ├── final_8features_state_IX.csv
│       ├── final_8features_state_V.csv
│       ├── merged_eis_capacity_state_I_norm.csv
│       ├── merged_eis_capacity_state_II_norm.csv
│       ├── merged_eis_capacity_state_III_norm.csv
│       ├── merged_eis_capacity_state_IV_norm.csv
│       ├── merged_eis_capacity_state_V_norm.csv
│       ├── merged_eis_capacity_state_VI_norm.csv
│       ├── merged_eis_capacity_state_VII_norm.csv
│       ├── merged_eis_capacity_state_VIII_norm.csv
│       ├── merged_eis_capacity_state_IX_norm.csv
│       └── merged_eis_capacity_state_V_IX_norm.csv
│
├── notebooks/
│   ├── models/               # Model training and evaluation
│   │   ├── 01_linear_regression.ipynb
│   │   ├── 02_random_forest.ipynb
│   │   ├── 03_rf_testing.ipynb
│   │   ├── 04_rf_validation.ipynb
│   │   ├── 05_xgboost_testing.ipynb
│   │   ├── 06_gpr_testing.ipynb
│   │   ├── 07_catboost_model.ipynb
│   │   ├── 08_stacking_ensemble.ipynb
│   │   └── 09_model_testing.ipynb
│   │
│   └── plots/                # Visualisation and SHAP analysis
│       └── 01_all_plots.ipynb
│
├── src/                      # Core Python scripts
│   ├── capacity_soh_cal.py
│   ├── merge_and_label_completedata.py
│   ├── merge_and_label_complete_extractiondata_state5.py
│   └── merge_and_label_data_stateV_norm.py
│
├── results/
│   ├── figures/              # All generated plots and images
│   └── tables/               # Result tables (model comparison etc.)
│
├── models/                   # Saved trained models
│   └── catboost_info/
│
├── requirements.txt
├── .gitignore
└── README.md
