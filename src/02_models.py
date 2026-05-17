"""
02_models.py
============
Negative binomial GLMs examining associations between health-related social
needs (HRSNs) and diabetes-related cardiovascular ED visit rates:

  - Model 1 : Any social need ever reported (binary)
  - Model 2 : Domain-specific needs ever reported (food, transportation,
              safety, utilities, housing instability, housing quality problems)
  - Model 3 : Social need burden (continuous: average positive domains per screening)

All models use a log(window_years) offset and HC0 robust standard errors.
Covariates: age group, sex, race/ethnicity, COVID-19-related ED visits.

Outputs (saved to ../outputs/):
  - Table4_model1_any_need.csv
  - Table5_model2_domain_specific.csv
  - Table6_model3_need_burden.csv
  - Figure2_IRR_forest_plot.png

Usage:
  python src/02_models.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

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
df["ever_hs_quality"] = (
    df[hs_quality_cols].fillna(0).sum(axis=1) >= 1
).astype(int)

# Social need burden (average positive domains per completed screening)
df["need_burden"] = df["need_rate"]


# ── Model fitting helper ───────────────────────────────────────────────────────
def fit_nb_glm(formula, data, offset_col, alpha=1.0, cov_type="HC0"):
    """Fit a Negative Binomial GLM with log(offset_col) as offset."""
    offset = np.log(data[offset_col].astype(float))
    model  = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=offset,
    )
    return model.fit(cov_type=cov_type)


def tidy_results(res, model_name):
    """Return a tidy coefficient table with IRRs and 95% CIs."""
    params = res.params
    conf   = res.conf_int()
    z_vals = getattr(res, "zvalues", getattr(res, "tvalues", None))

    out = pd.DataFrame({
        "Model":       model_name,
        "Term":        params.index,
        "B":           params.values.round(3),
        "SE":          res.bse.values.round(3),
        "IRR":         np.exp(params.values).round(3),
        "95% CI low":  np.exp(conf[0].values).round(3),
        "95% CI high": np.exp(conf[1].values).round(3),
        "z-value":     (pd.Series(z_vals).values.round(3)
                        if z_vals is not None else np.nan),
        "P value":     res.pvalues.values,
    })
    # Format P value
    out["P value"] = out["P value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )
    # Drop intercept row
    out = out[~out["Term"].str.lower().isin(["intercept", "const"])].copy()
    return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Any social need
# ══════════════════════════════════════════════════════════════════════════════
formula_m1 = """
    ED_window ~ ever_any_need
              + C(age4_cat, Treatment(reference="Adult (19–44 years)"))
              + C(gender,   Treatment(reference="Female"))
              + C(race,     Treatment(reference="Other race identified"))
              + COVID_ED_window
"""
df_m1 = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["ED_window", "window_years", "ever_any_need",
                       "age4_cat", "gender", "race", "COVID_ED_window"])
      .copy()
)
print(f"\n[Model 1] n = {len(df_m1):,}")
res_m1 = fit_nb_glm(formula_m1, df_m1, offset_col="window_years")

t4       = tidy_results(res_m1, "Model 1")
out_path = os.path.join(OUTPUT_DIR, "Table4_model1_any_need.csv")
t4.to_csv(out_path, index=False)
print(f"[Table 4] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Domain-specific social needs
# ══════════════════════════════════════════════════════════════════════════════
formula_m2 = """
    ED_window ~ housing_instability + ever_hs_quality
              + ever_fd + ever_tr + ever_sf + ever_ut
              + C(age4_cat, Treatment(reference="Adult (19–44 years)"))
              + C(gender,   Treatment(reference="Female"))
              + C(race,     Treatment(reference="Other race identified"))
              + COVID_ED_window
"""
df_m2 = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["ED_window", "window_years", "housing_instability",
                       "ever_hs_quality", "ever_fd", "ever_tr", "ever_sf",
                       "ever_ut", "age4_cat", "gender", "race", "COVID_ED_window"])
      .copy()
)
print(f"[Model 2] n = {len(df_m2):,}")
res_m2 = fit_nb_glm(formula_m2, df_m2, offset_col="window_years")

t5       = tidy_results(res_m2, "Model 2")
out_path = os.path.join(OUTPUT_DIR, "Table5_model2_domain_specific.csv")
t5.to_csv(out_path, index=False)
print(f"[Table 5] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Social need burden (continuous)
# ══════════════════════════════════════════════════════════════════════════════
formula_m3 = """
    ED_window ~ need_burden
              + C(age4_cat, Treatment(reference="Adult (19–44 years)"))
              + C(gender,   Treatment(reference="Female"))
              + C(race,     Treatment(reference="Other race identified"))
              + COVID_ED_window
"""
df_m3 = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["ED_window", "window_years", "need_burden",
                       "age4_cat", "gender", "race", "COVID_ED_window"])
      .copy()
)
print(f"[Model 3] n = {len(df_m3):,}")
res_m3 = fit_nb_glm(formula_m3, df_m3, offset_col="window_years")

t6       = tidy_results(res_m3, "Model 3")
out_path = os.path.join(OUTPUT_DIR, "Table6_model3_need_burden.csv")
t6.to_csv(out_path, index=False)
print(f"[Table 6] Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Forest plot: adjusted IRRs across all three models
# ══════════════════════════════════════════════════════════════════════════════

# Extract point estimates from fitted models (reproducible from data)
def extract_irr(res, term):
    """Return (IRR, CI_lo, CI_hi, p) for a given term."""
    conf = res.conf_int()
    irr  = float(np.exp(res.params[term]))
    lo   = float(np.exp(conf.loc[term, 0]))
    hi   = float(np.exp(conf.loc[term, 1]))
    p    = float(res.pvalues[term])
    return irr, lo, hi, p

plot_rows = [
    # label, irr, ci_lo, ci_hi, p_value, section
    ("Any Social Need",
     *extract_irr(res_m1, "ever_any_need"),
     "Model 1"),
    ("Housing Instability",
     *extract_irr(res_m2, "housing_instability"),
     "Model 2"),
    ("Housing Quality Problems",
     *extract_irr(res_m2, "ever_hs_quality"),
     "Model 2"),
    ("Food Insecurity",
     *extract_irr(res_m2, "ever_fd"),
     "Model 2"),
    ("Transportation Needs",
     *extract_irr(res_m2, "ever_tr"),
     "Model 2"),
    ("Safety Needs",
     *extract_irr(res_m2, "ever_sf"),
     "Model 2"),
    ("Utilities Needs",
     *extract_irr(res_m2, "ever_ut"),
     "Model 2"),
    ("Social Need Burden\n(per 1-unit increase)",
     *extract_irr(res_m3, "need_burden"),
     "Model 3"),
]

y_vals = [11, 9, 8, 7, 6, 5, 4, 2]

fig, ax = plt.subplots(figsize=(11, 8))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Alternating row shading
for i, y in enumerate(y_vals):
    if i % 2 == 0:
        ax.axhspan(y - 0.45, y + 0.45, color="#e8f0f7", alpha=0.6, zorder=0)

# Section header bands
for y_top, label in [
    (12.0, "Model 1: Any Social Need"),
    (10.2, "Model 2: Domain-Specific Social Needs"),
    (3.2,  "Model 3: Social Need Burden"),
]:
    ax.axhspan(y_top - 0.42, y_top + 0.42, color="#1a3c5e", alpha=0.12, zorder=0)
    ax.text(0.42, y_top, label, va="center", ha="left",
            fontsize=9.5, fontweight="bold", color="#1a3c5e", zorder=3)

# Plot each row
for (label, irr, lo, hi, p, _section), y in zip(plot_rows, y_vals):
    sig   = p < 0.05
    color = "#1a3c5e" if sig else "#8aa8c2"

    ax.plot([lo, hi], [y, y], color=color, lw=2.0, zorder=2, solid_capstyle="butt")
    for x_cap in [lo, hi]:
        ax.plot([x_cap, x_cap], [y - 0.15, y + 0.15], color=color, lw=1.8, zorder=2)
    ax.scatter(irr, y, s=90, color=color, marker="s",
               zorder=4, edgecolors="white", linewidths=0.8)

    lines = label.split("\n")
    if len(lines) == 2:
        ax.text(0.42, y + 0.22, lines[0], va="center", ha="left", fontsize=9, color="black")
        ax.text(0.42, y - 0.22, lines[1], va="center", ha="left", fontsize=8.5,
                color="#555555", style="italic")
    else:
        ax.text(0.42, y, label, va="center", ha="left", fontsize=9, color="black")

    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.text(7.8, y, f"{irr:.2f} ({lo:.2f}–{hi:.2f})  {p_str}",
            va="center", ha="right", fontsize=8.8, color="black", fontfamily="monospace")

# Reference line at IRR = 1
ax.axvline(1.0, color="#333333", lw=1.2, ls="-", zorder=1)

ax.set_xscale("log")
ax.set_xlim(0.4, 8.5)
ax.set_ylim(0.8, 13.5)
ax.set_xticks([0.5, 1.0, 2.0, 4.0])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis="x", labelsize=9)
ax.set_xlabel("Incidence Rate Ratio (IRR)", fontsize=10, labelpad=8)
ax.set_yticks([])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#aaaaaa")

ax.text(0.42, 13.3, "Exposure",
        fontsize=10, fontweight="bold", va="center", color="black")
ax.text(7.8, 13.3, "IRR (95% CI)   p-value",
        fontsize=10, fontweight="bold", va="center", ha="right",
        color="black", fontfamily="monospace")

ax.set_title(
    "Adjusted Incidence Rate Ratios for Health-Related Social Needs\n"
    "and Diabetes-Related Cardiovascular ED Visits",
    fontsize=11.5, fontweight="bold", pad=12, color="#111111"
)

sig_patch = mpatches.Patch(facecolor="#1a3c5e", label="Statistically significant (p < 0.05)")
ns_patch  = mpatches.Patch(facecolor="#8aa8c2", label="Not significant (p ≥ 0.05)")
ax.legend(handles=[sig_patch, ns_patch], loc="lower right",
          fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
          bbox_to_anchor=(1.0, 0.0))

fig.text(
    0.08, 0.01,
    "Note: All models adjusted for age group, sex, race/ethnicity, and COVID-19–related ED visits.\n"
    "Model 1: any social need vs none.  Model 2: domain-specific needs (simultaneously adjusted).\n"
    "Model 3: social need burden = average number of positive social need domains per completed screening.",
    fontsize=7.5, color="#555555", va="bottom"
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig_path = os.path.join(OUTPUT_DIR, "Figure2_IRR_forest_plot.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n[Figure 2] Saved → {fig_path}")

print("\n✓ All models and Figure 2 complete.")
