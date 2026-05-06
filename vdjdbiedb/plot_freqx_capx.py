import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
freq_min = 10
cap=100
INPUT_CSV = 'parsed_data_freq10_cap100.csv'
OUT_FIG   = 'freq10_cap100_distribution.png'

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, header=None, names=['id', 'pmhc', 'tcr', 'label'])
df['peptide'] = df['pmhc'].str.split('_').str[0]
df['hla']     = df['pmhc'].str.split('_HLA-').str[1]

freq = df['peptide'].value_counts().sort_values(ascending=False)
n_pep   = len(freq)
n_total = len(df)
n_cap   = (freq == cap).sum()
n_under = (freq < cap).sum()

print(f"Total peptides : {n_pep}")
print(f"Total records  : {n_total}")
print(f"cap=x reached: {n_cap}")
print(f"under cap      : {n_under}")

# ── 플롯 ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Per-peptide TCR count bar chart
ax = axes[0]
bar_colors = ['#2ca02c' if c == cap else '#1f77b4' for c in freq.values]
ax.bar(range(n_pep), freq.values, color=bar_colors, alpha=0.85, edgecolor='white')
ax.axhline(cap, color='red', linestyle='--', linewidth=1.3, label=f'cap={cap}')
ax.set_xticks(range(n_pep))
ax.set_xticklabels(freq.index, rotation=90, fontsize=6.5)
ax.set_xlabel('Peptide', fontsize=10)
ax.set_ylabel('Number of TCRs', fontsize=10)
ax.set_title(f'(a) Per-peptide TCR count  (n={n_pep} peptides, total={n_total})',
             fontsize=11)
ax.legend(handles=[
    Patch(color='#2ca02c', label=f'cap={cap} (n={n_cap})'),
    Patch(color='#1f77b4', label=f'under cap (n={n_under})'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.3,
               label=f'cap={cap} line'),
], fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# (c) HLA allele distribution
ax3 = axes[1]
hla_freq = df.groupby('hla')['peptide'].nunique().sort_values(ascending=False)
cmap = plt.cm.Set3(np.linspace(0, 1, len(hla_freq)))
bars3 = ax3.bar(range(len(hla_freq)), hla_freq.values,
                color=cmap, alpha=0.85, edgecolor='white')
ax3.set_xticks(range(len(hla_freq)))
ax3.set_xticklabels(hla_freq.index, rotation=45, ha='right', fontsize=8)
ax3.set_xlabel('HLA Allele', fontsize=10)
ax3.set_ylabel('Number of Peptides', fontsize=10)
ax3.set_title('(b) Peptide count per HLA allele', fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, hla_freq.values):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.15,
             str(val), ha='center', va='bottom', fontsize=8)
plt.suptitle(
    f'Dataset Distribution  (freq >= {freq_min}, cap = {cap})\n'
    f'{n_pep} peptides  |  {n_total} positive TCRs',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
print(f"Saved -> {OUT_FIG}")
plt.close()