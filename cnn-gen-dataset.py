import pandas as pd
import numpy as np

# 파일 경로 설정
cnn_cate_filter_path = "./cnn-cate-filter.csv"
fonts_all_ko_path = "/Users/doheyonkim/data/fontbox/ttfs/2025_font/fonts_all_ko.csv"
output_path = "cnn-cate_filter_merged.csv"

# 1. CSV 파일 읽기
cnn_cate_filter_df = pd.read_csv(cnn_cate_filter_path)
fonts_all_ko_df = pd.read_csv(fonts_all_ko_path)

# 2. 공백이나 비어있는 값들을 처리하고 ID 필드를 정수형으로 변환
# cnn_cate_filter_df의 font_id 처리
cnn_cate_filter_df['font_id'] = pd.to_numeric(cnn_cate_filter_df['font_id'], errors='coerce')
cnn_cate_filter_df = cnn_cate_filter_df.dropna(subset=['font_id'])  # font_id가 NaN인 행 제거
cnn_cate_filter_df['font_id'] = cnn_cate_filter_df['font_id'].astype(int)

# fonts_all_ko_df의 FontID 처리
# 먼저 공백 문자를 NaN으로 변환
fonts_all_ko_df['FontID'] = fonts_all_ko_df['FontID'].replace(' ', np.nan)
fonts_all_ko_df['FontID'] = pd.to_numeric(fonts_all_ko_df['FontID'], errors='coerce')
fonts_all_ko_df = fonts_all_ko_df.dropna(subset=['FontID'])  # FontID가 NaN인 행 제거
fonts_all_ko_df['FontID'] = fonts_all_ko_df['FontID'].astype(int)

# 3. 두 데이터프레임을 font_id를 기준으로 병합
# left join을 사용하여 cnn_cate_filter_df의 모든 행을 유지
merged_df = pd.merge(
    cnn_cate_filter_df,
    fonts_all_ko_df,
    left_on='font_id',
    right_on='FontID',
    how='left'
)

# FontID 열은 중복되므로 제거
if 'FontID' in merged_df.columns:
    merged_df = merged_df.drop('FontID', axis=1)

# 4. FileName과 FilePath가 비어있는 행 제외
# NaN 값과 빈 문자열('') 모두 고려하여 필터링
merged_df = merged_df.dropna(subset=['FileName', 'FilePath'])  # NaN 값 제외
merged_df = merged_df[(merged_df['FileName'] != '') & (merged_df['FilePath'] != '')]  # 빈 문자열 제외

# 5. 결과를 CSV 파일로 저장
merged_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"병합된 파일이 {output_path}에 저장되었습니다.")
print(f"총 {len(merged_df)} 개의 행이 처리되었습니다.")
print("처음 5개 행:")
print(merged_df.head())