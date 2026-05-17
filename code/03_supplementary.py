"""
03_supplementary.py
===================
Supplementary analyses:

  - eTable 1 : Co-occurrence of social needs among participants reporting
               housing instability or housing quality problems
  - eTable 2 : Pairwise Pearson correlations and VIFs among social need
               variables included in Model 2
  - eTable 3 : Input parameters for national cost-impact projections
               (reproduced from published sources; no individual-level data needed)
  - eTable 4 : Projected preventable ED visits and cost savings under two
               social need reduction scenarios (PARF framework)

Outputs (saved to ../outputs/):
  - eTable1_housing_cooccurrence.csv
  - eTable2_correlations_vif.csv
  - eTable3_projection_parameters.csv
  - eTable4_parf_projections.csv

Usage:
  python src/03_supplementary.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "AHC_DT_CVD.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Derived variables (mirror 02_models.py) ────────────────────────────────────
df["housing_instability"] = (
    (df["cn_hs_1_no_stdy_hsng"].fillna(0) >= 1) |
    (df["cn_hs_1_hsng_unsure"].fillna(0)  >= 1)
).astype(int)

hs_quality_cols = [
    "cn_hsqual_water_leaks", "cn_hsqual_bug_infestation",
    "cn_hsqual_lead_paint_pipes", "cn_hsqual_smoke_det_not_wkg",
    "cn_hsqual_mold", "cn_hsqual_inadq_heat",
]
df["housing_quality_problems"] = (
    df[hs_quality_cols].fillna(0).sum(axis=1) >= 1
).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# eTABLE 1 — Co-occurrence of social needs within housing need groups
# ══════════════════════════════════════════════════════════════════════════════
hi_group  = df[df["housing_instability"]      == 1]
hq_group  = df[df["housing_quality_problems"] == 1]

cooccur_vars = [
    ("ever_fd",                  "Food insecurity"),
    ("ever_tr",                  "Transportation needs"),
    ("ever_ut",                  "Utilities needs"),
    ("housing_quality_problems", "Housing quality problems"),   # within HI group
    ("housing_instability",      "Housing instability"),        # within HQ group
    ("ever_sf",                  "Interpersonal safety"),
]

rows_e1 = []

# Housing instability group
for var, label in cooccur_vars:
    if var == "housing_instability":
        continue  # skip self-reference
    n   = int(hi_group[var].fillna(0).sum())
    pct = 100 * n / len(hi_group)
    rows_e1.append({
        "Housing Need Group":     f"Housing instability (n={len(hi_group):,})",
        "Co-occurring Social Need": label,
        "n":                      n,
        "% Within Group":         f"{pct:.1f}%",
    })

# Housing quality problems group
for var, label in cooccur_vars:
    if var == "housing_quality_problems":
        continue  # skip self-reference
    n   = int(hq_group[var].fillna(0).sum())
    pct = 100 * n / len(hq_group)
    rows_e1.append({
        "Housing Need Group":       f"Housing quality problems (n={len(hq_group):,})",
        "Co-occurring Social Need": label,
        "n":                        n,
        "% Within Group":           f"{pct:.1f}%",
    })

etable1  = pd.DataFrame(rows_e1)
out_path = os.path.join(OUTPUT_DIR, "eTable1_housing_cooccurrence.csv")
etable1.to_csv(out_path, index=False)
print(f"\n[eTable 1] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# eTABLE 2 — Pairwise Pearson correlations and VIFs (Model 2 predictors)
# ══════════════════════════════════════════════════════════════════════════════
need_vars = [
    "housing_instability",
    "housing_quality_problems",
    "ever_fd",
    "ever_tr",
    "ever_sf",
    "ever_ut",
]

nice_names = {
    "housing_instability":      "Housing instability (HI)",
    "housing_quality_problems": "Housing quality problems (HQ)",
    "ever_fd":                  "Food insecurity (FD)",
    "ever_tr":                  "Transportation needs (TR)",
    "ever_sf":                  "Interpersonal safety (SF)",
    "ever_ut":                  "Utilities needs (UT)",
}

# Drop rows with any missing in these variables
df_vif = df[need_vars].dropna().copy()
print(f"[eTable 2] VIF sample: n={len(df_vif):,}")

# Pairwise Pearson correlations
corr_matrix = df_vif.rename(columns=nice_names).corr(method="pearson").round(3)
corr_matrix = corr_matrix.where(
    ~pd.DataFrame(
        np.eye(len(need_vars), dtype=bool),
        index=corr_matrix.index, columns=corr_matrix.columns
    )
)  # replace diagonal with NaN (self-correlation → "—")

# VIF via OLS
X     = add_constant(df_vif)
vif_vals = {
    nice_names[col]: round(variance_inflation_factor(X.values, i + 1), 3)
    for i, col in enumerate(need_vars)
}

corr_out = corr_matrix.copy()
corr_out["VIF"] = pd.Series(vif_vals)

corr_out = corr_out.fillna("—").reset_index().rename(columns={"index": "Social Need Variable"})
out_path = os.path.join(OUTPUT_DIR, "eTable2_correlations_vif.csv")
corr_out.to_csv(out_path, index=False)
print(f"[eTable 2] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# eTABLE 3 — Input parameters for national cost-impact projections
# These values are sourced from published external data (see column "Source").
# ══════════════════════════════════════════════════════════════════════════════
etable3_rows = [
    {
        "Parameter":    "National diabetes-related ED visits (2021)",
        "Value":        "16,500,000",
        "Source":       "CDC National Diabetes Statistics Report, 2026",
        "Notes":        "Any-listed diabetes diagnosis",
    },
    {
        "Parameter":    "Cardiovascular proportion",
        "Value":        "11.8%",
        "Source":       "HCUP Statistical Brief #167 (2010)",
        "Notes":        "Sum of four first-listed CVD diagnosis categories (chest pain 5.6%, CHF 3.3%, cerebrovascular 1.5%, AMI 1.4%); conservative lower bound",
    },
    {
        "Parameter":    "Government payer proportion (Medicare + Medicaid)",
        "Value":        "68.7%",
        "Source":       "HCUP Statistical Brief #167 (2010)",
        "Notes":        "Medicare 53.7%, Medicaid 15.0%; applied to restrict to study-relevant population",
    },
    {
        "Parameter":    "Medicare average ED facility cost",
        "Value":        "$1,040",
        "Source":       "HCUP / CMS, 2021",
        "Notes":        "Facility cost only; excludes professional fees",
    },
    {
        "Parameter":    "Medicaid average ED facility cost",
        "Value":        "$600",
        "Source":       "HCUP / CMS, 2021",
        "Notes":        "Facility cost only; excludes professional fees",
    },
    {
        "Parameter":    "Social need prevalence (P)",
        "Value":        "60.4%",
        "Source":       "Current study",
        "Notes":        "Proportion of cohort reporting ≥1 core social need",
    },
    {
        "Parameter":    "Adjusted incidence rate ratio (IRR)",
        "Value":        "1.31",
        "Source":       "Current study — Model 1",
        "Notes":        "Any social need vs none; adjusted for age, sex, race/ethnicity, COVID-19 ED visits",
    },
]

etable3  = pd.DataFrame(etable3_rows)
out_path = os.path.join(OUTPUT_DIR, "eTable3_projection_parameters.csv")
etable3.to_csv(out_path, index=False)
print(f"\n[eTable 3] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# eTABLE 4 — PARF projections under social need reduction scenarios
#
# Population Attributable Risk Fraction (PARF):
#   PARF = P × (IRR − 1) / [P × (IRR − 1) + 1]
# where P = social need prevalence, IRR = adjusted incidence rate ratio.
# ══════════════════════════════════════════════════════════════════════════════

# Fixed parameters (from eTable 3)
NATIONAL_DIABETES_ED   = 16_500_000          # total national diabetes-related ED visits
CVD_PROPORTION         = 0.118               # cardiovascular proportion
GOVT_PAYER_PROPORTION  = 0.687               # Medicare + Medicaid share
MEDICARE_SHARE         = 0.537 / (0.537 + 0.150)   # ~78.2% of govt-insured visits
MEDICAID_SHARE         = 1 - MEDICARE_SHARE         # ~21.8%
MEDICARE_COST          = 1_040               # $ per visit
MEDICAID_COST          = 600                 # $ per visit
IRR                    = 1.31                # Model 1 IRR (any social need)
P_BASELINE             = 0.604               # social need prevalence in cohort

# Derived fixed values
CVD_ED_NATIONAL        = NATIONAL_DIABETES_ED * CVD_PROPORTION          # ~1,947,000
GOVT_CVD_ED            = CVD_ED_NATIONAL * GOVT_PAYER_PROPORTION         # ~1,337,589
WEIGHTED_COST          = MEDICARE_SHARE * MEDICARE_COST + MEDICAID_SHARE * MEDICAID_COST  # ~$944

print(f"\n[eTable 4] National Medicare/Medicaid diabetes-related CVD ED visits: {GOVT_CVD_ED:,.0f}")
print(f"           Payer-weighted average ED cost: ${WEIGHTED_COST:,.2f}")

def parf(p, irr):
    """Population Attributable Risk Fraction."""
    return p * (irr - 1) / (p * (irr - 1) + 1)

# Baseline
parf_base = parf(P_BASELINE, IRR)

# Scenarios
scenarios = [
    ("10% reduction in social need prevalence",
     P_BASELINE * 0.90,
     f"{P_BASELINE*100:.1f}% → {P_BASELINE*0.90*100:.1f}%"),
    ("30% reduction in social need prevalence",
     P_BASELINE * 0.70,
     f"{P_BASELINE*100:.1f}% → {P_BASELINE*0.70*100:.1f}%"),
]

rows_e4 = []
for scenario_label, p_new, prev_change in scenarios:
    parf_new       = parf(p_new, IRR)
    parf_reduction = parf_base - parf_new
    visits_prevented = GOVT_CVD_ED * parf_reduction
    cost_savings     = visits_prevented * WEIGHTED_COST

    rows_e4.append({
        "Scenario":                     scenario_label,
        "Prevalence change":            prev_change,
        "Baseline PARF":                f"{parf_base*100:.2f}%",
        "New PARF":                     f"{parf_new*100:.2f}%",
        "PARF reduction (pp)":          round(parf_reduction * 100, 2),
        "Visits prevented per year (n)": f"~{round(visits_prevented / 10) * 10:,.0f}",
        "Estimated cost savings":       f"~${cost_savings / 1e6:.1f} million/year",
    })
    print(f"  {scenario_label}:")
    print(f"    PARF: {parf_base*100:.2f}% → {parf_new*100:.2f}%  "
          f"(Δ {parf_reduction*100:.2f} pp)")
    print(f"    Visits prevented: ~{visits_prevented:,.0f}  |  "
          f"Cost savings: ~${cost_savings/1e6:.1f}M/year")

etable4  = pd.DataFrame(rows_e4)
out_path = os.path.join(OUTPUT_DIR, "eTable4_parf_projections.csv")
etable4.to_csv(out_path, index=False)
print(f"\n[eTable 4] Saved → {out_path}")

print("\n✓ All supplementary tables complete.")
