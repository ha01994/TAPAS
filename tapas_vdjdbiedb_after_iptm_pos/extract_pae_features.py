import os
import glob
import numpy as np
import pandas as pd

def extract_pae_features(pae_dir):
    """
    지정된 디렉토리 내의 모든 _interface_pae.npy 파일을 읽어
    TabPFN 학습용 통계적 피처를 추출합니다.
    """
    # 폴더 내의 모든 pae npy 파일 경로 찾기
    file_pattern = os.path.join(pae_dir, '*_interface_pae.npy')
    pae_files = glob.glob(file_pattern)
    
    features_list = []
    
    for file_path in pae_files:
        # 파일명에서 ID 추출 (예: v0, v1000 등)
        file_name = os.path.basename(file_path)
        sample_id = file_name.split('_interface_pae')[0]
        
        # PAE 매트릭스 로드
        pae_matrix = np.load(file_path)
        
        # 매트릭스에 NaN이나 Inf가 있는지 확인하고 처리 (안전성 보장)
        if np.isnan(pae_matrix).any() or np.isinf(pae_matrix).any():
            pae_matrix = np.nan_to_num(pae_matrix, nan=31.75, posinf=31.75) # AF3 PAE 최대값 근처로 처리
            
        # 2D 매트릭스를 1D 배열로 변환하여 통계 계산
        pae_flat = pae_matrix.flatten()
        
        # ---------------------------------------------------------
        # 1. 기본 통계 피처 (Mean, Min, Max, Std)
        # ---------------------------------------------------------
        mean_pae = np.mean(pae_flat)
        min_pae = np.min(pae_flat)
        max_pae = np.max(pae_flat)
        std_pae = np.std(pae_flat)
        
        # ---------------------------------------------------------
        # 2. 퍼센타일 및 중간값 (Outlier-robust features)
        # ---------------------------------------------------------
        median_pae = np.median(pae_flat)
        p10_pae = np.percentile(pae_flat, 10)
        p90_pae = np.percentile(pae_flat, 90)
        
        # ---------------------------------------------------------
        # 3. 임계값 기반 비율 피처 (Threshold-based fractions)
        # ---------------------------------------------------------
        # 결합이 매우 확실하다고 판단되는 영역의 비율 (예: 5Å 이하)
        frac_less_5 = np.mean(pae_flat < 5.0)
        
        # 결합이 사실상 끊어졌다고 판단되는 영역의 비율 (예: 15Å 이상)
        frac_greater_15 = np.mean(pae_flat > 15.0)
        
        # ---------------------------------------------------------
        # 4. 비대칭성 (Asymmetry) - 선택 사항
        # ---------------------------------------------------------
        # *주의*: 이 부분은 pae_matrix가 정사각형(Square) 매트릭스일 때만 작동합니다.
        # 즉, [TCR+Pep] x [TCR+Pep] 전체 인터페이스가 저장된 경우.
        # 직사각형 매트릭스(TCR x Pep)만 저장되었다면 이 부분은 제외하세요.
        asymmetry = 0.0
        if pae_matrix.shape[0] == pae_matrix.shape[1]:
            # |PAE_ij - PAE_ji| 의 평균
            asymmetry = np.mean(np.abs(pae_matrix - pae_matrix.T))
            
        # 결과를 딕셔너리로 저장
        features = {
            'sample_id': sample_id,
            'pae_mean': mean_pae,
            #'pae_min': min_pae,
            'pae_max': max_pae,
            'pae_std': std_pae,
            'pae_median': median_pae,
            'pae_p10': p10_pae,
            'pae_p90': p90_pae,
            'pae_frac_lt_5': frac_less_5,
            'pae_frac_gt_15': frac_greater_15,
            'pae_asymmetry': asymmetry  # 매트릭스가 정방형일 경우 주석 해제
        }
        
        features_list.append(features)
        
    # 데이터프레임으로 변환
    df_features = pd.DataFrame(features_list)
    return df_features

# ==========================================
# 실행 부분
# ==========================================
# 실제 폴더 경로로 지정해주세요.
pae_directory = './pae_matrix_vdjdbiedb' 

# 피처 추출
pae_features_df = extract_pae_features(pae_directory)

# TabPFN에 입력하기 위해 다른 피처(ipTM 등) 데이터프레임과 병합할 수 있도록
# CSV 파일로 저장해 둡니다.
pae_features_df.to_csv('pae_feat_vdjdbiedb.csv', index=False)

print("PAE 피처 추출 완료! 10개 샘플 미리보기:")
print(pae_features_df.head(10))
