import pickle
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PICKLE_FILE = 'shap_values_rs_fold0.pkl'
OUT_BAR     = 'shap_top15_bar_rs_fold0.png'
OUT_BEE     = 'shap_top15_beeswarm_rs_fold0.png'
TOP_N       = 15

# ── 로드 ──────────────────────────────────────────────────────
with open(PICKLE_FILE, 'rb') as f:
    shap_values = pickle.load(f)

# ── 이름 변경 ─────────────────────────────────────────────────
rename_map = {
    'iptm_tcrpmhc'  : 'ipTM (TCR-pMHC)',
    'iptm_mean'     : 'ipTM (mean)',
    'global_plddt'  : 'Global pLDDT',
    'pdockq'        : 'pDockQ',
    'avgipae_pmhc'  : 'iPAE (pMHC)',
    'avgipae_tcr'   : 'iPAE (TCR)',
    'pdockq2_pmhc'  : 'pDockQ2 (pMHC)',
    'pdockq2_tcr'   : 'pDockQ2 (TCR)',
    'cdr1_A'        : 'CDR1α pLDDT',
    'cdr2_A'        : 'CDR2α pLDDT',
    'cdr3_A'        : 'CDR3α pLDDT',
    'cdr1_B'        : 'CDR1β pLDDT',
    'cdr2_B'        : 'CDR2β pLDDT',
    'cdr3_B'        : 'CDR3β pLDDT',
    'ESM_Peptide'   : 'ESM: Peptide',
    'ESM_CDRA1'     : 'ESM: CDR1α',
    'ESM_CDRA2'     : 'ESM: CDR2α',
    'ESM_CDRA3'     : 'ESM: CDR3α',
    'ESM_CDRB1'     : 'ESM: CDR1β',
    'ESM_CDRB2'     : 'ESM: CDR2β',
    'ESM_CDRB3'     : 'ESM: CDR3β',
}
shap_values.feature_names = [rename_map.get(f, f) for f in shap_values.feature_names]

# ── Top 15 선택 ───────────────────────────────────────────────
mean_abs_shap   = np.abs(shap_values.values).mean(axis=0)
top_idx         = np.argsort(mean_abs_shap)[::-1][:TOP_N]
shap_values_top = shap_values[:, top_idx]

# ── Bar plot ──────────────────────────────────────────────────
top_names = [shap_values.feature_names[i] for i in top_idx]
top_vals  = mean_abs_shap[top_idx]

top_names_plot = top_names[::-1]
top_vals_plot  = top_vals[::-1]

fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.barh(top_names_plot, top_vals_plot, color='crimson', alpha=0.85)

for bar, val in zip(bars, top_vals_plot):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'+{val:.3f}', va='center', fontsize=8)

ax.set_xlabel('mean(|SHAP value|)', fontsize=11)
ax.set_title('Feature Importance — Conf+PAE (RS)', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_BAR, dpi=300, bbox_inches='tight')
plt.close()
print(f"저장 → {OUT_BAR}")

# ── Beeswarm plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values_top, max_display=TOP_N, show=False)
plt.title('SHAP Beeswarm — Conf+PAE (RS)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_BEE, dpi=300, bbox_inches='tight')
plt.close()
print(f"저장 → {OUT_BEE}")