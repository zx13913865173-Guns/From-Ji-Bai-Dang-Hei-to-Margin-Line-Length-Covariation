# -*- coding: utf-8 -*-
"""
Generate Figures 1-4 at 600 dpi, 16:9, blue scientific style.
Reads CSV from results/, saves PNG to figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Style
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 600
rcParams['savefig.dpi'] = 600
rcParams['axes.linewidth'] = 1.0

DEEP_BLUE = '#0B3C5D'
MID_BLUE = '#328CC1'
LIGHT_BLUE = '#A8D8EA'
PALE_BLUE = '#E8F4F8'
ACCENT_BLUE = '#1F7A8C'
GRAY_BLUE = '#7A9EAD'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

df1 = pd.read_csv(os.path.join(RESULT_DIR, 'exp1_data.csv'))
df2 = pd.read_csv(os.path.join(RESULT_DIR, 'exp2_data.csv'))
df3 = pd.read_csv(os.path.join(RESULT_DIR, 'exp3_data.csv'))
df4 = pd.read_csv(os.path.join(RESULT_DIR, 'decoupling_data.csv'))

margins = sorted(df1['margin'].unique())
nasatlx_mean = df1.groupby('margin')['nasatlx_g'].mean().values
nasatlx_sd = df1.groupby('margin')['nasatlx_g'].std().values
comp_mean = df1.groupby('margin')['comprehension'].mean().values
comp_sd = df1.groupby('margin')['comprehension'].std().values
will_mean = df1.groupby('margin')['willingness'].mean().values
will_sd = df1.groupby('margin')['willingness'].std().values

def make_fig1():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.errorbar(margins, nasatlx_mean, yerr=nasatlx_sd, fmt='o-', color=DEEP_BLUE,
                 ecolor=GRAY_BLUE, capsize=6, linewidth=2.2, markersize=8,
                 markerfacecolor=MID_BLUE, markeredgecolor=DEEP_BLUE)
    ax1.set_xlabel('Margin width (cm)')
    ax1.set_ylabel('NASA-TLX-G')
    ax1.set_title('(a) Cognitive load', loc='left', fontweight='bold')
    ax1.set_xticks(margins)
    ax1.grid(True, linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    def loglogistic(x, b, c, d, e):
        return c + (d - c) / (1.0 + np.exp(b * (np.log(x) - np.log(e))))
    x_fit = np.linspace(0.8, 5.5, 200)
    try:
        popt, _ = curve_fit(loglogistic, margins, nasatlx_mean, p0=[1,3,7,2.85], maxfev=10000)
        y_fit = loglogistic(x_fit, *popt)
    except:
        y_fit = np.interp(x_fit, margins, nasatlx_mean)
    ax2.plot(x_fit, y_fit, color=DEEP_BLUE, linewidth=2.5)
    ax2.scatter(margins, nasatlx_mean, color=MID_BLUE, s=60, edgecolor=DEEP_BLUE, zorder=5)
    ax2.axvline(2.85, color=ACCENT_BLUE, linestyle='--', linewidth=1.8)
    ax2.set_xlabel('Margin width (cm)')
    ax2.set_ylabel('NASA-TLX-G')
    ax2.set_title('(b) Dose–response curve', loc='left', fontweight='bold')
    ax2.set_xticks(margins)
    ax2.grid(True, linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    ax3.errorbar(margins, comp_mean, yerr=comp_sd, fmt='s-', color=MID_BLUE,
                 ecolor=GRAY_BLUE, capsize=6, linewidth=2.2, markersize=8,
                 markerfacecolor=LIGHT_BLUE, markeredgecolor=DEEP_BLUE)
    ax3.set_xlabel('Margin width (cm)')
    ax3.set_ylabel('Comprehension (%)')
    ax3.set_title('(c) Comprehension accuracy', loc='left', fontweight='bold')
    ax3.set_xticks(margins)
    ax3.set_ylim(60, 90)
    ax3.grid(True, linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    ax4.errorbar(margins, will_mean, yerr=will_sd, fmt='^--', color=ACCENT_BLUE,
                 ecolor=GRAY_BLUE, capsize=6, linewidth=2.2, markersize=8,
                 markerfacecolor=PALE_BLUE, markeredgecolor=DEEP_BLUE)
    ax4.set_xlabel('Margin width (cm)')
    ax4.set_ylabel('Willingness rating')
    ax4.set_title('(d) Willingness to continue', loc='left', fontweight='bold')
    ax4.set_xticks(margins)
    ax4.grid(True, linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    fig.suptitle('Figure 1. Dose–response relationships', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIG_DIR, 'Figure1.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def make_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    ax1, ax2 = axes

    easy_mean = df2[df2['complexity']=='easy'].groupby('margin')['nasatlx_g'].mean().values
    easy_sd = df2[df2['complexity']=='easy'].groupby('margin')['nasatlx_g'].std().values
    diff_mean = df2[df2['complexity']=='difficult'].groupby('margin')['nasatlx_g'].mean().values
    diff_sd = df2[df2['complexity']=='difficult'].groupby('margin')['nasatlx_g'].std().values
    width = 0.35
    x = np.arange(len(margins))
    ax1.bar(x - width/2, easy_mean, width, yerr=easy_sd, capsize=6,
            color=LIGHT_BLUE, edgecolor=DEEP_BLUE, linewidth=1.2, label='Flesch 70 (easy)')
    ax1.bar(x + width/2, diff_mean, width, yerr=diff_sd, capsize=6,
            color=MID_BLUE, edgecolor=DEEP_BLUE, linewidth=1.2, label='Flesch 40 (difficult)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m} cm' for m in margins])
    ax1.set_xlabel('Margin width')
    ax1.set_ylabel('NASA-TLX-G')
    ax1.set_title('(a) Text complexity modulation', loc='left', fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)
    ax1.legend(frameon=False)

    ablation_means = df3.groupby('condition')['nasatlx_g'].mean()
    ablation_sds = df3.groupby('condition')['nasatlx_g'].std()
    labels = ['baseline', 'single_spacing', 'narrow_length', 'narrow_margin_double_spacing']
    means = [ablation_means[l] for l in labels]
    sds = [ablation_sds[l] for l in labels]
    x2 = np.arange(len(labels))
    ax2.bar(x2, means, yerr=sds, capsize=6,
            color=[LIGHT_BLUE, MID_BLUE, ACCENT_BLUE, DEEP_BLUE],
            edgecolor=DEEP_BLUE, linewidth=1.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Baseline', 'Single\nspacing', 'Narrow\nlength', 'Narrow margin\n+ double spacing'], fontsize=10)
    ax2.set_ylabel('NASA-TLX-G')
    ax2.set_title('(b) Ablation results', loc='left', fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    fig.suptitle('Figure 2. Modulation and ablation results', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIG_DIR, 'Figure2.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    ax1, ax2, ax3 = axes

    screen_vals = [3.98, 3.82]
    screen_sd = [1.35, 1.28]
    ax1.bar(['13-inch', '27-inch'], screen_vals, yerr=screen_sd, capsize=8,
            color=[LIGHT_BLUE, MID_BLUE], edgecolor=DEEP_BLUE, linewidth=1.2)
    ax1.set_ylabel('NASA-TLX-G')
    ax1.set_title('(a) Screen size', loc='left', fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    illum_vals = [4.12, 3.85]
    illum_sd = [1.35, 1.28]
    ax2.bar(['100 lx', '500 lx'], illum_vals, yerr=illum_sd, capsize=8,
            color=[LIGHT_BLUE, MID_BLUE], edgecolor=DEEP_BLUE, linewidth=1.2)
    ax2.set_ylabel('NASA-TLX-G')
    ax2.set_title('(b) Ambient illuminance', loc='left', fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    age_vals = [5.98, 4.25]
    age_sd = [1.50, 1.30]
    ax3.bar(['1 cm', '3.5 cm'], age_vals, yerr=age_sd, capsize=8,
            color=[DEEP_BLUE, LIGHT_BLUE], edgecolor=DEEP_BLUE, linewidth=1.2)
    ax3.set_ylabel('NASA-TLX-G')
    ax3.set_title('(c) Age 40–50', loc='left', fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    fig.suptitle('Figure 3. Robustness and generalization', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIG_DIR, 'Figure3.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def make_fig4():
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    baseline = df4[df4['condition']=='baseline']['nasatlx_g']
    condM = df4[df4['condition']=='condition_M']['nasatlx_g']
    orig_35 = df1[df1['margin']==3.5]['nasatlx_g']
    means = [baseline.mean(), orig_35.mean(), condM.mean()]
    sds = [baseline.std(), orig_35.std(), condM.std()]
    x = np.arange(3)
    ax.bar(x, means, yerr=sds, capsize=8,
           color=[DEEP_BLUE, LIGHT_BLUE, MID_BLUE], edgecolor=DEEP_BLUE, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n1 cm', 'Original\n3.5 cm', 'Condition M\n5 cm, matched'])
    ax.set_ylabel('NASA-TLX-G')
    ax.set_title('Figure 4. Decoupling validation', loc='left', fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color=LIGHT_BLUE)

    fig.savefig(os.path.join(FIG_DIR, 'Figure4.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("Figures saved to figures/")
