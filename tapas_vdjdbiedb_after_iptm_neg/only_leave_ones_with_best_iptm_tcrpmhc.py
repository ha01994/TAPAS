import pandas as pd

def main(input_file, output_file):
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: '{input_file}' 파일을 찾을 수 없습니다. 파일이 같은 폴더에 있는지 확인해주세요.")
        return

    # 2. pdb_id별로 iptm_tcrpmhc 값이 가장 큰 행들만 추출
    # idxmax() 함수를 사용하여 각 그룹에서 최대값을 가지는 인덱스를 구합니다.
    best_models_idx = df.groupby('pdb_id')['iptm_tcrpmhc'].idxmax()
    df_best = df.loc[best_models_idx]

    # (선택) pdb_id를 기준으로 정렬 (필요시)
    # df_best = df_best.sort_values(by='pdb_id').reset_index(drop=True)

    # 3. 결과를 새로운 CSV 파일로 저장
    df_best.to_csv(output_file, index=False)
    
    print(len(df_best))
    
    print(f"성공적으로 최고 모델만 추출하여 '{output_file}'에 저장했습니다.")

    
    
if __name__ == '__main__':
    
    input_file = 'results_model_quality_metrics.csv'
    output_file = 'results_vdjdbiedb_neg_iptm_filtered_best.csv'
    main(input_file, output_file)
    
    for f in range(3):
        input_file = f'results_model_quality_metrics_{f}.csv'
        output_file = f'results_model_quality_metrics_{f}_best.csv'
        main(input_file, output_file)

    