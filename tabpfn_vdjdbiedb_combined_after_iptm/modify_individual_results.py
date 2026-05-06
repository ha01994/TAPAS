"""
`test_zeroshot_individual.py`가 저장한 zeroshot-by-feature 결과 CSV를 후처리한다.

- 수치 컬럼을 지정 자릿수(기본 3)로 반올림한 뒤 통합 파일을 갱신(덮어쓰기)한다.
- RS / SS 행을 각각 `zeroshot_individual_by_feature_rs.csv`,
  `zeroshot_individual_by_feature_ss.csv`로 저장한다.
  이 두 파일에는 `feature`, `auc_mean` 컬럼만 포함한다.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def round_numeric_except(
    df: pd.DataFrame,
    exclude: frozenset[str],
    ndigits: int,
) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(ndigits)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Post-process test_zeroshot_individual.py output '
            '(round metrics, split RS/SS to auc_mean-only files).'
        )
    )
    parser.add_argument(
        '--input',
        default='results_auc/zeroshot_individual_by_feature.csv',
        help='입력 CSV (test_zeroshot_individual.py 출력; feature, split, auc_*, auc01_*)',
    )
    parser.add_argument(
        '--out-combined',
        default='results_auc/zeroshot_individual_by_feature.csv',
        help='반올림된 통합 CSV 저장 경로 (기본: 입력과 동일, 덮어쓰기)',
    )
    parser.add_argument(
        '--out-rs',
        default='results_auc/zeroshot_individual_by_feature_rs.csv',
        help='split==rs 행만: feature, auc_mean',
    )
    parser.add_argument(
        '--out-ss',
        default='results_auc/zeroshot_individual_by_feature_ss.csv',
        help='split==ss 행만: feature, auc_mean',
    )
    parser.add_argument(
        '--decimals',
        type=int,
        default=3,
        help='소수 자릿수 (기본 3)',
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'ERROR: 입력 파일이 없습니다: {args.input}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if len(df) == 0:
        print(f'ERROR: 입력 CSV에 행이 없습니다: {args.input}', file=sys.stderr)
        sys.exit(1)

    if 'split' not in df.columns:
        print("ERROR: 'split' 컬럼이 필요합니다 (rs / ss).", file=sys.stderr)
        sys.exit(1)
    if 'auc_mean' not in df.columns:
        print("ERROR: 'auc_mean' 컬럼이 필요합니다.", file=sys.stderr)
        sys.exit(1)
    if 'feature' not in df.columns:
        print("ERROR: 'feature' 컬럼이 필요합니다.", file=sys.stderr)
        sys.exit(1)

    exclude = frozenset({'feature', 'split'})
    rounded = round_numeric_except(df, exclude, args.decimals)
    float_fmt = f'%.{args.decimals}f'

    os.makedirs(os.path.dirname(args.out_combined) or '.', exist_ok=True)
    rounded.to_csv(args.out_combined, index=False, float_format=float_fmt)

    rs_mask = rounded['split'].astype(str).str.lower() == 'rs'
    ss_mask = rounded['split'].astype(str).str.lower() == 'ss'
    if not rs_mask.any():
        print("WARNING: split=='rs' 행이 없습니다.", file=sys.stderr)
    if not ss_mask.any():
        print("WARNING: split=='ss' 행이 없습니다.", file=sys.stderr)

    split_cols = ['feature', 'auc_mean']
    rs_df = rounded.loc[rs_mask, split_cols].copy()
    ss_df = rounded.loc[ss_mask, split_cols].copy()

    os.makedirs(os.path.dirname(args.out_rs) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ss) or '.', exist_ok=True)
    rs_df.to_csv(args.out_rs, index=False, float_format=float_fmt)
    ss_df.to_csv(args.out_ss, index=False, float_format=float_fmt)

    print(f'Wrote (rounded {args.decimals} dp): {args.out_combined}')
    print(f'Wrote RS ({len(rs_df)} rows): {args.out_rs}')
    print(f'Wrote SS ({len(ss_df)} rows): {args.out_ss}')


if __name__ == '__main__':
    main()
