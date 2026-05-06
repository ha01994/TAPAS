import pandas as pd
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────────────────
INPUT_CSV  = 'parsed_data_downsampled_final.csv'
FREQ_MIN   = 10    # 펩타이드 최소 frequency
CAP        = 100   # 펩타이드당 최대 샘플 수
OUT_CSV    = f'parsed_data_freq{FREQ_MIN}_cap{CAP}.csv'
RANDOM_SEED = 42

# ── 로드 ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, header=None, names=['id', 'pmhc', 'tcr', 'label'])
df['peptide'] = df['pmhc'].str.split('_').str[0]
print(f"원본 shape: {df.shape}")
print(f"원본 unique peptides: {df['peptide'].nunique()}")

# ── Step 1: frequency >= freq_min 펩타이드만 선택 ───────────────────────────────────
freq = df['peptide'].value_counts()
valid_peps = freq[freq >= FREQ_MIN].index.tolist()
df_filtered = df[df['peptide'].isin(valid_peps)].copy()

print(f"\n[Step 1] frequency >= {FREQ_MIN} 필터링")
print(f"  유효 펩타이드: {len(valid_peps)}개")
print(f"  필터링 후 shape: {df_filtered.shape}")

# ── Step 2: 펩타이드당 최대 cap개로 다운샘플링 ───────────────────────────────
sampled_rows = []
for pep in valid_peps:
    rows = df_filtered[df_filtered['peptide'] == pep]
    if len(rows) > CAP:
        rows = rows.sample(n=CAP, random_state=RANDOM_SEED)
    sampled_rows.append(rows)
df_sampled = pd.concat(sampled_rows, ignore_index=True)

print(f"\n[Step 2] 펩타이드당 최대 {CAP}개로 cap")
print(f"  최종 shape: {df_sampled.shape}")

# ── 결과 확인 ─────────────────────────────────────────────────────────────────
final_freq = df_sampled['peptide'].value_counts().sort_values(ascending=False)
print(f"\n최종 펩타이드별 샘플 수:")
print(f"{'peptide':<20} {'count':>6}")
print("-" * 28)
for pep, cnt in final_freq.items():
    print(f"{pep:<20} {cnt:>6}")

print(f"\n요약:")
print(f"  총 펩타이드 수:  {df_sampled['peptide'].nunique()}")
print(f"  총 레코드 수:    {len(df_sampled)}")
print(f"  cap 100 도달:   {(final_freq == CAP).sum()}개 펩타이드")
print(f"  cap 미만:       {(final_freq < CAP).sum()}개 펩타이드")

# ── 저장 (헤더 없이 원본 형식 유지) ──────────────────────────────────────────
df_sampled[['id', 'pmhc', 'tcr', 'label']].to_csv(OUT_CSV, index=False, header=False)
print(f"\n저장 → {OUT_CSV}")