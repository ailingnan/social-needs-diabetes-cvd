"""
01_tables.py
============
Descriptive statistics for:
  - Table 1: Patient characteristics (all / any social need / no social need)
  - Table 2: Social needs prevalence and ED utilization during the screening window
  - Table 3: Unadjusted diabetes-related cardiovascular ED visit rates by social need

Outputs (saved to ../outputs/):
  - Table1_patient_characteristics.csv
  - Table1_chi_square_tests.csv
  - Table2_social_needs_utilization.csv
  - Table3_unadjusted_rates.csv

Usage:
  python src/01_tables.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "AHC_DT_CVD.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Derived variables ──────────────────────────────────────────────────────────

# Housing instability: no steady housing OR worried about losing housing
df["housing_instability"] = (
    (df["cn_hs_1_no_stdy_hsng"].fillna(0) >= 1) |
    (df["cn_hs_1_hsng_unsure"].fillna(0)  >= 1)
).astype(int)

# Housing quality problems: any quality item endorsed at least once
hs_quality_cols = [
    "cn_hsqual_water_leaks", "cn_hsqual_bug_infestation",
    "cn_hsqual_lead_paint_pipes", "cn_hsqual_smoke_det_not_wkg",
    "cn_hsqual_mold", "cn_hsqual_inadq_heat",
]
df["housing_quality_problems"] = (
    df[hs_quality_cols].fillna(0).sum(axis=1) >= 1
).astype(int)

# Any housing need (not mutually exclusive with each other)
df["any_housing_need"] = (
    (df["housing_instability"] == 1) | (df["housing_quality_problems"] == 1)
).astype(int)

# Screening count category
def screening_count_cat(x):
    if pd.isna(x): return np.nan
    return "1" if x == 1 else ("2" if x == 2 else "3+")
df["screening_count_cat"] = df["screening_count"].apply(screening_count_cat)

# Screening window length category (years)
def window_year_cat(x):
    if pd.isna(x): return np.nan
    return "1" if x <= 1 else ("2" if x <= 2 else "3+")
df["window_year_cat"] = df["window_years"].apply(window_year_cat)

# Binary utilization indicators
df["any_diab_cvd_ed"] = (df["ED_window"].fillna(0) >= 1).astype(int)
df["any_covid_ed"]    = (df["COVID_ED_window"].fillna(0) >= 1).astype(int)

# Annualised ED visit rate (visits per screening year)
df["ED_rate"] = df["ED_window"] / df["window_years"].replace({0: np.nan})

# Subgroups by social need status
any_need = df[df["ever_any_need"] == 1].copy()
no_need  = df[df["ever_any_need"] == 0].copy()
N, N_any, N_no = len(df), len(any_need), len(no_need)
print(f"  Any social need: n={N_any:,}  |  No social need: n={N_no:,}")


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt_n_pct(series, level):
    s = series.dropna()
    n = (s == level).sum()
    pct = 100 * n / len(s) if len(s) > 0 else np.nan
    return f"{int(n):,} ({pct:.2f})"

def n_pct_binary(series):
    n   = int(series.fillna(0).sum())
    pct = 100 * n / len(series) if len(series) > 0 else np.nan
    return f"{n:,} ({pct:.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Patient characteristics
# ══════════════════════════════════════════════════════════════════════════════
col_all = f"All samples [N={N:,}]"
col_any = f"Any Social Need [n={N_any:,}]"
col_no  = f"No Social Need [n={N_no:,}]"

def add_section(rows, title):
    rows.append({"Characteristic": title, col_all: "", col_any: "", col_no: ""})

def add_categorical(rows, var, label, level_order):
    rows.append({"Characteristic": f"  {label}", col_all: "", col_any: "", col_no: ""})
    for lv in level_order:
        rows.append({
            "Characteristic": f"    {lv}",
            col_all: fmt_n_pct(df[var],       lv),
            col_any: fmt_n_pct(any_need[var], lv),
            col_no:  fmt_n_pct(no_need[var],  lv),
        })

rows = []
add_section(rows, "Demographic characteristics")
add_categorical(rows, "age4_cat", "Age group", [
    "Minor (<19 years)", "Adult (19–44 years)",
    "Middle-aged (45–64 years)", "Aged (≥65 years)",
])
add_categorical(rows, "gender", "Gender", ["Female", "Male"])
add_categorical(rows, "race", "Race/ethnicity", [
    "White (NH)", "Black/AfAm (NH)", "Hispanic",
    "Asian/PacIslander", "Other race identified",
])
add_section(rows, "Screening Information")
add_categorical(rows, "screening_count_cat", "Screening count", ["1", "2", "3+"])
add_categorical(rows, "window_year_cat", "Screening window length (years)", ["1", "2", "3+"])

table1   = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "Table1_patient_characteristics.csv")
table1.to_csv(out_path, index=False)
print(f"\n[Table 1] Saved → {out_path}")

# Chi-square tests (for footnote b)
def chi2_any_vs_none(var, levels):
    counts = np.array([
        [(any_need[var] == lv).sum(), (no_need[var] == lv).sum()]
        for lv in levels
    ])
    counts = counts[counts.sum(axis=1) > 0]
    chi2, p, dof, _ = chi2_contingency(counts)
    return chi2, p, dof

chi_tests = [
    ("age4_cat",            "Age group",                 ["Minor (<19 years)", "Adult (19–44 years)", "Middle-aged (45–64 years)", "Aged (≥65 years)"]),
    ("gender",              "Gender",                    ["Female", "Male"]),
    ("race",                "Race/ethnicity",            ["White (NH)", "Black/AfAm (NH)", "Hispanic", "Asian/PacIslander", "Other race identified"]),
    ("screening_count_cat", "Screening count",           ["1", "2", "3+"]),
    ("window_year_cat",     "Screening window (years)",  ["1", "2", "3+"]),
]

chi_rows = []
for var, label, levels in chi_tests:
    chi2, p, dof = chi2_any_vs_none(var, levels)
    chi_rows.append({
        "Variable":    label,
        "Chi-square":  round(chi2, 3),
        "df":          dof,
        "P value":     "<0.001" if p < 0.001 else f"{p:.3f}",
    })

chi_table = pd.DataFrame(chi_rows)
out_path  = os.path.join(OUTPUT_DIR, "Table1_chi_square_tests.csv")
chi_table.to_csv(out_path, index=False)
print(f"[Table 1 χ²] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Social needs prevalence and ED utilization
# ══════════════════════════════════════════════════════════════════════════════
col = f"All samples [N={N:,}]"
rows2 = [
    {"Characteristic": "Social needs prevalence",                    col: ""},
    {"Characteristic": "Any social need",                            col: n_pct_binary(df["ever_any_need"])},
    {"Characteristic": "Domain-specific needs ever reported",        col: ""},
    {"Characteristic": "  Food",                                     col: n_pct_binary(df["ever_fd"])},
    {"Characteristic": "  Transportation",                           col: n_pct_binary(df["ever_tr"])},
    {"Characteristic": "  Utilities",                                col: n_pct_binary(df["ever_ut"])},
    {"Characteristic": "  Safety",                                   col: n_pct_binary(df["ever_sf"])},
    {"Characteristic": "  Any housing needs",                        col: n_pct_binary(df["any_housing_need"])},
    {"Characteristic": "    Housing instability",                    col: n_pct_binary(df["housing_instability"])},
    {"Characteristic": "    Housing quality problems",               col: n_pct_binary(df["housing_quality_problems"])},
    {"Characteristic": "ED utilization during the screening window", col: ""},
    {"Characteristic": "Diabetes-related cardiovascular ED visits",  col: ""},
    {"Characteristic": "  Any visit",                                col: n_pct_binary(df["any_diab_cvd_ed"])},
    {"Characteristic": "  No visit",                                 col: f"{(df['any_diab_cvd_ed']==0).sum():,} ({100*(df['any_diab_cvd_ed']==0).mean():.2f})"},
    {"Characteristic": "COVID-19-related ED visits",                 col: ""},
    {"Characteristic": "  Any visit",                                col: n_pct_binary(df["any_covid_ed"])},
    {"Characteristic": "  No visit",                                 col: f"{(df['any_covid_ed']==0).sum():,} ({100*(df['any_covid_ed']==0).mean():.2f})"},
]

table2   = pd.DataFrame(rows2)
out_path = os.path.join(OUTPUT_DIR, "Table2_social_needs_utilization.csv")
table2.to_csv(out_path, index=False)
print(f"[Table 2] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Unadjusted ED visit rates by social need (Mann–Whitney U)
# ══════════════════════════════════════════════════════════════════════════════
need_vars = [
    ("ever_any_need",            "Ever any social need"),
    ("ever_fd",                  "Ever food needs"),
    ("ever_tr",                  "Ever transportation needs"),
    ("ever_sf",                  "Ever safety needs"),
    ("ever_ut",                  "Ever utilities needs"),
    ("housing_instability",      "Ever housing instability"),
    ("housing_quality_problems", "Ever housing quality problems"),
]

rows3 = []
for var, label in need_vars:
    pos  = df.loc[df[var] == 1, "ED_rate"].dropna()
    neg  = df.loc[df[var] == 0, "ED_rate"].dropna()
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    rows3.append({
        "Social need":               label,
        "n (positive)":              len(pos),
        "ED rate (positive)":        round(pos.mean(), 4),
        "n (negative)":              len(neg),
        "ED rate (negative)":        round(neg.mean(), 4),
        "Mann–Whitney U statistic":  round(stat, 2),
        "P value":                   "<0.001" if p < 0.001 else f"{p:.3f}",
    })

table3   = pd.DataFrame(rows3)
out_path = os.path.join(OUTPUT_DIR, "Table3_unadjusted_rates.csv")
table3.to_csv(out_path, index=False)
print(f"[Table 3] Saved → {out_path}")

print("\n✓ All descriptive tables complete.")
