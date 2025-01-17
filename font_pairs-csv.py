import argparse
import os
import csv
import uuid
def create_font_pairs_csv():

    # insert input paramter 2 path

    # rootPath = os.path.dirname(os.path.abspath(__file__))
    # 생성을 하기 위한 경로
    # target_path (추론하고 싶은 폰트들이 있는 경로)  -- 숫자_폰트명.ttf
    # origin_path (추론할때 사용해야하는 폰트 경로)   -- 숫자_폰트명.ttf
    # lora_path (lora 파일이 저장되어야할 경로)

    # input parameter
    # python font_pairs-csv.py --csv_file=/home/user/data
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='/Users/doheyonkim/data/fontbox/ttfs', help='Path to CSV file containing dataset paths')
    args = parser.parse_args()
    rootPath = args.csv_file

    # 경로 설정
    origin_path = rootPath + '/sandoll-origin'
    target_path = rootPath + '/sandoll-target'
    lora_path = rootPath + '/lora'
    train_path = rootPath + '/train'
    sumnail_path = rootPath + '/sumnail'

    # mkdir folder
    os.makedirs(origin_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(lora_path, exist_ok=True)
    os.makedirs(sumnail_path, exist_ok=True)

    # 파일 리스트 가져오기
    origin_files = os.listdir(origin_path)
    target_files = os.listdir(target_path)

    # 숫자를 키로 하는 딕셔너리 생성
    origin_dict = {}
    target_dict = {}

    # origin 파일 정리
    for file in origin_files:
        if file.startswith('.'):  # 숨김 파일 제외
            continue
        num = int(file.split('-')[0])
        origin_dict[num] = os.path.join(origin_path, file)

    # target 파일 정리
    for file in target_files:
        if file.startswith('.'):  # 숨김 파일 제외
            continue
        num = int(file.split('-')[0])
        target_dict[num] = os.path.join(target_path, file)

    # 매칭된 쌍만 찾기
    pairs = []

    for num in sorted(set(origin_dict.keys()) & set(target_dict.keys())):
        uuid_str = str(uuid.uuid4())
        # 0: target, 1: origin, 2: uuid, 3: train_path, 4: lora_path, 5: sumnail_path
        pairs.append([target_dict[num], origin_dict[num], uuid_str, train_path + '/' + uuid_str, lora_path + '/' + uuid_str, sumnail_path + '/' + uuid_str])

    # CSV 파일로 저장
    with open('font_pairs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(pairs)

    print(f"Created CSV file with {len(pairs)} pairs")


# 실행
create_font_pairs_csv()