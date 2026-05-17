# Health-Related Social Needs and Diabetes-Related Cardiovascular ED Utilization

> **Nan A, Liu B, Fu MR, Wiest D, Qiu Z.**  
> *Health-Related Social Needs and Emergency Department Utilizations for Diabetes-Related Cardiovascular Disease: Evidence from Southern New Jersey.*  

---

## Overview

This repository contains the analysis code for a retrospective cohort study examining associations between health-related social needs (HRSNs) and diabetes-related cardiovascular emergency department (ED) utilization. Data were drawn from the **Camden Coalition Health Information Exchange (CCHIE)**, spanning 2017–2022, and include Medicare and Medicaid beneficiaries who completed at least one Accountable Health Communities (AHC) social needs screening (N = 21,037).

Three negative binomial generalized linear models (with log-offset for screening window duration) were estimated:

| Model | Exposure |
|-------|----------|
| Model  1 | Any social need ever reported (binary) |
| Model  2 | Domain-specific needs ever reported (food, transportation, safety, utilities, housing instability, housing quality problems) |
| Model  3 | Social need burden (continuous: average number of positive domains per screening) |

All models adjusted for age group, sex, race/ethnicity, and COVID-19-related ED visits during the screening window.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md           ← Data access instructions
└── src/
    ├── 01_tables.py            → Tables 1–3 (descriptive statistics)
    ├── 02_models.py            → Models 1–3, Tables 4–6, and Figure 2 (forest plot)
    └── 03_supplementary.py         → eTables 1–4 and PARF calculations
```

---

## Data Availability

**The analysis dataset is not publicly shared** due to data use agreement restrictions with the Camden Coalition.

To request the analysis-ready dataset (`AHC_DT_CVD.csv`), please contact:

> **Bowen Liu, PhD**  
> School of Science and Engineering, University of Missouri–Kansas City  
> 📧 bowen.liu@umkc.edu

> Please include a brief description of your intended use in your request.
---

## Reproducing the Analysis

### Prerequisites

- Python 3.12

### 1. Clone the repository

```bash
git clone https://github.com/ailingnan/social-needs-diabetes-cvd.git
cd social-needs-diabetes-cvd
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Obtain the data

Request `AHC_DT_CVD.csv` from the contact above and place it in the `data/` directory:

```
data/
└── AHC_DT_CVD.csv
```

### 4. Run the scripts

Run the three scripts in order from the repository root:

```bash
python src/01_tables.py
python src/02_models.py
python src/03_supplementary.py
```

All outputs are saved automatically to `outputs/`.

| Step | Script | Description | Outputs |
|------|--------|-------------|---------|
| 1 | `src/01_tables.py` | Descriptive statistics, social needs prevalence, unadjusted ED rates | `Table1_patient_characteristics.csv`, `Table1_chi_square_tests.csv`, `Table2_social_needs_utilization.csv`, `Table3_unadjusted_rates.csv` |
| 2 | `src/02_models.py` | Negative binomial GLMs (Models 1–3), result tables, forest plot | `Table4_model1_any_need.csv`, `Table5_model2_domain_specific.csv`, `Table6_model3_need_burden.csv`, `Figure2_IRR_forest_plot.png` |
| 3 | `src/03_supplementary.py` | Housing co-occurrence, VIF/correlation matrix, PARF projections | `eTable1_housing_cooccurrence.csv`, `eTable2_correlations_vif.csv`, `eTable3_projection_parameters.csv`, `eTable4_parf_projections.csv` |

---

## Variable Reference

| Variable | Description |
|----------|-------------|
| `ED_window` | Count of diabetes-related cardiovascular ED visits during the screening window |
| `window_years` | Duration of the screening window in years (used as model offset) |
| `ever_any_need` | Binary: any core social need identified at ≥1 screening |
| `ever_fd` | Binary: food insecurity ever reported |
| `ever_tr` | Binary: transportation need ever reported |
| `ever_sf` | Binary: interpersonal safety need ever reported |
| `ever_ut` | Binary: utilities need ever reported |
| `need_rate` | Social need burden (total positive domain screens ÷ total screenings) |
| `housing_instability` | Derived in `02_models.py`: no steady housing OR worried about losing housing |
| `ever_hs_quality` | Derived in `02_models.py`: any housing quality problem ever reported |
| `COVID_ED_window` | Count of COVID-19-related ED visits during the screening window (covariate) |
| `age4_cat` | Age group at first screening: Minor (<19), Adult (19–44), Middle-aged (45–64), Aged (≥65) |
| `gender` | Sex (Female / Male) |
| `race` | Race/ethnicity (White NH, Black/AfAm NH, Hispanic, Asian/PacIslander, Other) |

---

## Citation

If you use this code, please cite the associated manuscript (citation details will be updated upon publication).

---

## License

To be determined upon publication.

---

## Contact

For questions about the analysis, please open a GitHub Issue or contact **Ailing Nan** at anzcc@umkc.edu.
