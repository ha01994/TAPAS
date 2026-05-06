import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = 'parsed_data_downsampled_final.csv'
FREQ_MIN   = 10    # minimum peptide frequency
CAP        = 100   # max samples per peptide
OUT_CSV    = f'parsed_data_freq{FREQ_MIN}_cap{CAP}.csv'
RANDOM_SEED = 42

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, header=None, names=['id', 'pmhc', 'tcr', 'label'])
df['peptide'] = df['pmhc'].str.split('_').str[0]
print(f"original shape: {df.shape}")
print(f"original unique peptides: {df['peptide'].nunique()}")

# ── Step 1: select peptides with frequency >= freq_min ───────────────────────────
freq = df['peptide'].value_counts()
valid_peps = freq[freq >= FREQ_MIN].index.tolist()
df_filtered = df[df['peptide'].isin(valid_peps)].copy()

print(f"\n[Step 1] frequency >= {FREQ_MIN} filtering")
print(f"  valid peptides: {len(valid_peps)}items")
print(f"  shape after filtering: {df_filtered.shape}")

# ── Step 2: downsample to max cap items per peptide ─────────────────────────────
sampled_rows = []
for pep in valid_peps:
    rows = df_filtered[df_filtered['peptide'] == pep]
    if len(rows) > CAP:
        rows = rows.sample(n=CAP, random_state=RANDOM_SEED)
    sampled_rows.append(rows)
df_sampled = pd.concat(sampled_rows, ignore_index=True)

print(f"\n[Step 2] Max per peptide: {CAP}items cap")
print(f"  final shape: {df_sampled.shape}")

# ── Check results ─────────────────────────────────────────────────────────────
final_freq = df_sampled['peptide'].value_counts().sort_values(ascending=False)
print(f"\nFinal sample counts per peptide:")
print(f"{'peptide':<20} {'count':>6}")
print("-" * 28)
for pep, cnt in final_freq.items():
    print(f"{pep:<20} {cnt:>6}")

print(f"\nSummary:")
print(f"  total peptides:    {df_sampled['peptide'].nunique()}")
print(f"  total records:     {len(df_sampled)}")
print(f"  reached cap=100:   {(final_freq == CAP).sum()}peptides")
print(f"  below cap:         {(final_freq < CAP).sum()}peptides")

# ── Save (keep original format without header) ───────────────────────────────
df_sampled[['id', 'pmhc', 'tcr', 'label']].to_csv(OUT_CSV, index=False, header=False)
print(f"\nSaved → {OUT_CSV}")