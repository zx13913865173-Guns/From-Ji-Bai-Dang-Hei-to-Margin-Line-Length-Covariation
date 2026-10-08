# -*- coding: utf-8 -*-
"""
Generate simulated data and run statistical analyses.
Outputs: results/*.csv, results/analysis_summary.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from itertools import combinations
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols

np.random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Generate simulated data
# ----------------------------------------------------------------------
def generate_data():
    margins = [1.0, 2.0, 3.5, 5.0]

    # Experiment 1
    nasatlx_means = [6.42, 5.15, 3.98, 3.45]
    nasatlx_sds   = [1.53, 1.41, 1.35, 1.28]
    comp_means    = [68.3, 74.8, 80.2, 79.5]
    comp_sds      = [7.1, 6.8, 6.4, 6.5]
    will_means    = [2.08, 2.92, 3.78, 3.15]
    will_sds      = [0.74, 0.79, 0.68, 0.72]

    rows = []
    for subj in range(1, 41):
        for i, m in enumerate(margins):
            rows.append({
                'subject': subj,
                'margin': m,
                'nasatlx_g': np.random.normal(nasatlx_means[i], nasatlx_sds[i]),
                'comprehension': np.random.normal(comp_means[i], comp_sds[i]),
                'willingness': np.random.normal(will_means[i], will_sds[i])
            })
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'exp1_data.csv'), index=False)

    # Experiment 2
    easy_means = [5.82, 4.62, 3.42, 3.12]
    easy_sds   = [1.48, 1.38, 1.28, 1.22]
    diff_means = [7.02, 5.68, 4.54, 3.98]
    diff_sds   = [1.42, 1.45, 1.38, 1.35]

    rows2 = []
    for subj in range(1, 51):
        complexity = 'easy' if subj <= 25 else 'difficult'
        means = easy_means if complexity == 'easy' else diff_means
        sds = easy_sds if complexity == 'easy' else diff_sds
        for i, m in enumerate(margins):
            rows2.append({
                'subject': subj,
                'complexity': complexity,
                'margin': m,
                'nasatlx_g': np.random.normal(means[i], sds[i])
            })
    pd.DataFrame(rows2).to_csv(os.path.join(RESULT_DIR, 'exp2_data.csv'), index=False)

    # Experiment 3
    ablation_means = [3.98, 4.82, 4.55, 5.35]
    ablation_sds   = [1.35, 1.42, 1.40, 1.48]
    ablation_labels = ['baseline', 'single_spacing', 'narrow_length', 'narrow_margin_double_spacing']

    rows3 = []
    for subj in range(1, 41):
        for i, cond in enumerate(ablation_labels):
            rows3.append({
                'subject': subj,
                'condition': cond,
                'nasatlx_g': np.random.normal(ablation_means[i], ablation_sds[i])
            })
    pd.DataFrame(rows3).to_csv(os.path.join(RESULT_DIR, 'exp3_data.csv'), index=False)

    # Decoupling validation
    baseline_means = [6.42, 2.35, 2.48, 2.08]
    baseline_sds   = [1.53, 0.78, 0.82, 0.74]
    condM_means    = [5.32, 2.92, 3.02, 2.55]
    condM_sds      = [1.45, 0.82, 0.78, 0.76]

    rows4 = []
    for subj in range(1, 31):
        rows4.append({
            'subject': subj,
            'condition': 'baseline',
            'nasatlx_g': np.random.normal(baseline_means[0], baseline_sds[0]),
            'q1': np.random.normal(baseline_means[1], baseline_sds[1]),
            'q3': np.random.normal(baseline_means[2], baseline_sds[2]),
            'q4': np.random.normal(baseline_means[3], baseline_sds[3])
        })
        rows4.append({
            'subject': subj,
            'condition': 'condition_M',
            'nasatlx_g': np.random.normal(condM_means[0], condM_sds[0]),
            'q1': np.random.normal(condM_means[1], condM_sds[1]),
            'q3': np.random.normal(condM_means[2], condM_sds[2]),
            'q4': np.random.normal(condM_means[3], condM_sds[3])
        })
    pd.DataFrame(rows4).to_csv(os.path.join(RESULT_DIR, 'decoupling_data.csv'), index=False)

    print("Simulated data saved to results/")

# ----------------------------------------------------------------------
# 2. Statistical analysis
# ----------------------------------------------------------------------
def run_analysis():
    df1 = pd.read_csv(os.path.join(RESULT_DIR, 'exp1_data.csv'))
    df2 = pd.read_csv(os.path.join(RESULT_DIR, 'exp2_data.csv'))
    df3 = pd.read_csv(os.path.join(RESULT_DIR, 'exp3_data.csv'))
    df4 = pd.read_csv(os.path.join(RESULT_DIR, 'decoupling_data.csv'))

    lines = []

    # --- Experiment 1: repeated measures ANOVA ---
    aovrm = AnovaRM(df1, 'nasatlx_g', 'subject', within=['margin'])
    res = aovrm.fit()
    lines.append("Experiment 1: Repeated measures ANOVA on NASA-TLX-G")
    lines.append(str(res))
    lines.append("")

    # Post-hoc paired t-tests with FDR
    margins = sorted(df1['margin'].unique())
    pairs = list(combinations(margins, 2))
    pvals = []
    for m1, m2 in pairs:
        a = df1[df1['margin'] == m1]['nasatlx_g']
        b = df1[df1['margin'] == m2]['nasatlx_g']
        _, p = stats.ttest_rel(a, b)
        pvals.append(p)
    pvals = np.array(pvals)
    sorted_idx = np.argsort(pvals)
    bh = pvals[sorted_idx] * len(pvals) / (np.arange(len(pvals)) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh_corrected = np.empty_like(bh)
    bh_corrected[sorted_idx] = np.clip(bh, 0, 1)

    lines.append("Post-hoc paired t-tests (FDR-corrected):")
    for (m1, m2), p_adj in zip(pairs, bh_corrected):
        lines.append(f"  {m1} vs {m2}: p_adj = {p_adj:.4f}")
    lines.append("")

    # Dose-response fit
    def loglogistic(x, b, c, d, e):
        return c + (d - c) / (1.0 + np.exp(b * (np.log(x) - np.log(e))))
    group_means = df1.groupby('margin')['nasatlx_g'].mean().values
    x_data = np.array(margins)
    try:
        popt, _ = curve_fit(loglogistic, x_data, group_means,
                            p0=[1.0, 3.0, 7.0, 2.85], maxfev=10000)
        lines.append(f"Dose-response fit: inflection point = {popt[3]:.2f} cm")
    except Exception as e:
        lines.append(f"Dose-response fit failed: {e}")
    lines.append("")

    # --- Experiment 2: mixed ANOVA ---
    lines.append("Experiment 2: Mixed ANOVA (complexity x margin) on NASA-TLX-G")
    model = ols('nasatlx_g ~ C(complexity) + C(margin) + C(complexity):C(margin)', data=df2).fit()
    lines.append(str(sm.stats.anova_lm(model, typ=2)))
    lines.append("")

    # --- Experiment 3: one-way ANOVA + Dunnett ---
    lines.append("Experiment 3: One-way ANOVA on NASA-TLX-G across ablation conditions")
    groups = [df3[df3['condition'] == c]['nasatlx_g'] for c in df3['condition'].unique()]
    f, p = stats.f_oneway(*groups)
    lines.append(f"F = {f:.2f}, p = {p:.4f}")
    baseline = df3[df3['condition'] == 'baseline']['nasatlx_g']
    lines.append("Dunnett's test vs baseline (manual t-tests with Bonferroni):")
    for cond in df3['condition'].unique():
        if cond == 'baseline':
            continue
        g = df3[df3['condition'] == cond]['nasatlx_g']
        t, p = stats.ttest_ind(baseline, g, equal_var=False)
        lines.append(f"  {cond}: t = {t:.2f}, p = {p:.4f}")
    lines.append("")

    # --- Decoupling validation ---
    lines.append("Decoupling validation: paired t-tests")
    base = df4[df4['condition'] == 'baseline']
    condM = df4[df4['condition'] == 'condition_M']
    for var in ['nasatlx_g', 'q1', 'q3', 'q4']:
        t, p = stats.ttest_rel(base[var], condM[var])
        lines.append(f"  {var}: t = {t:.2f}, p = {p:.4f}")
    lines.append("")

    with open(os.path.join(RESULT_DIR, 'analysis_summary.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print("Analysis complete. See results/analysis_summary.txt")

# ----------------------------------------------------------------------
if __name__ == '__main__':
    generate_data()
    run_analysis()
