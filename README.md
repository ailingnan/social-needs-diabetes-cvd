# Health-Related Social Needs and Diabetes-Related Cardiovascular ED Utilization

> **Nan A, Liu B, Fu MR, Wiest D, Qiu Z.**  
> *Health-Related Social Needs and Emergency Department Utilizations for Diabetes-Related Cardiovascular Disease: Evidence from Southern New Jersey.*  
> [Manuscript under review]

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
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
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

### 4. Run the notebooks

Launch Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

| Step | Notebook | Description | Output |
|------|----------|-------------|--------|
| 1 | `notebooks/_Table.ipynb` | Descriptive statistics, social needs prevalence, unadjusted ED rates | `Table1_patient_characteristics.csv`, `Table1_chi_square_tests.csv`, `Table2_*.csv`, `Table3_*.csv` |
| 2 | `notebooks/_Model.ipynb` | Negative binomial GLMs (Models 1–3), forest plot, VIF check | `Table3_results.csv`, `IRR_forest_plot_updated.png` |

> **Note:** The notebooks expect the data file at `../data/AHC_DT_CVD.csv` relative to the `notebooks/` directory. Update the `pd.read_csv(...)` path if your working directory differs.

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
| `ever_hs` | Binary: any housing need ever reported |
| `need_rate` | Social need burden (total positive domain screens ÷ total screenings) |
| `housing_instability` | Derived in `_Model.ipynb`: no steady housing OR worried about losing housing |
| `ever_hs_quality` | Derived in `_Model.ipynb`: any housing quality problem ever reported |
| `COVID_ED_window` | Count of COVID-19-related ED visits during the screening window (covariate) |
| `age4_cat` | Age group at first screening: Minor (<19), Adult (19–44), Middle-aged (45–64), Aged (≥65) |
| `gender` | Sex (Female / Male) |
| `race` | Race/ethnicity (White NH, Black/AfAm NH, Hispanic, Asian/PacIslander, Other) |

---

## Key Findings

- Patients reporting **any social need** had a **31% higher rate** of diabetes-related cardiovascular ED visits (IRR 1.31, 95% CI 1.03–1.66, *p* = .029)
- **Food insecurity** (IRR 1.48), **interpersonal safety needs** (IRR 2.79), and **transportation barriers** (IRR 1.36) were each independently associated with higher ED visit rates
- **Social need burden** showed a dose-response relationship (IRR 1.19 per 1-unit increase, *p* = .002)

---

## Software and Statistical Methods

- **Language:** Python 3.12
- **Key packages:** `pandas`, `numpy`, `statsmodels` 0.14.4, `scipy`, `matplotlib`
- **Models:** Negative binomial GLM with log-link and log(window duration) offset (`statsmodels.formula.api.glm` + `sm.families.NegativeBinomial`)
- **Standard errors:** Heteroskedasticity-consistent (HC0 robust)
- **Unadjusted comparisons:** Mann–Whitney U test

---

## Citation

If you use this code, please cite the associated manuscript (citation details will be updated upon publication).

---

## License

To be determined upon publication.

---

## Contact

For questions about the analysis, please open a GitHub Issue or contact **Ailing Nan** at anzcc@umkc.edu.
