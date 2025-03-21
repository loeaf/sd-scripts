import csv
import pandas as pd
import os
import argparse
import subprocess
from pathlib import Path
import json
import requests
import toml
import uuid
import shutil

GOOGLE_CHAT_WEBHOOK = "https://chat.googleapis.com/v1/spaces/AAAABLzMLsI/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=lwSnIq9XvRuT56A9BRh1xEuE-wU8vzqny_skSrTMIio"


def send_google_chat_message(message):
    """Google Chat으로 메시지를 보내는 함수"""
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {'text': message}
    response = requests.post(GOOGLE_CHAT_WEBHOOK, headers=headers, data=json.dumps(data))
    return response.status_code


def process_image_set(image_dir):
    """
    이미지 파일들의 이름을 UUID로 변경하고 동일한 이름의 텍스트 파일을 생성합니다.

    Args:
        image_dir (str): 이미지 파일들이 있는 디렉토리 경로

    Returns:
        str: 처리된 이미지를 포함하는 디렉토리 경로
    """
    # 이미지 확장자 목록
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    results = []

    # 결과를 저장할 디렉토리 생성
    output_dir = os.path.join(image_dir, f"processed_{int(Path(image_dir).stat().st_mtime)}")
    os.makedirs(output_dir, exist_ok=True)

    # 결과를 저장할 CSV 파일 경로
    output_csv = os.path.join(output_dir, 'font_pairs.csv')

    # LoRA 저장 경로 생성
    lora_dir = os.path.join(output_dir, 'lora')
    os.makedirs(lora_dir, exist_ok=True)

    # 디렉토리 내의 모든 파일 순회
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)

        # 파일인지 확인하고 이미지 확장자인지 확인
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # UUID 생성
            file_uuid = str(uuid.uuid4())

            # 원본 파일 확장자 가져오기
            _, ext = os.path.splitext(filename)

            # 새 파일명 (UUID + 원본 확장자)
            new_filename = f"{file_uuid}{ext}"
            new_file_path = os.path.join(output_dir, new_filename)

            # 동일한 이름의 텍스트 파일 경로
            txt_filename = f"{file_uuid}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)

            # 이미지 파일 복사
            shutil.copy2(file_path, new_file_path)

            # 동일한 이름의 빈 텍스트 파일 생성
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f"Original filename: {filename}\n")
                txt_file.write(f"UUID: {file_uuid}\n")

            # LoRA 모델이 저장될 경로
            lora_path = os.path.join(lora_dir, file_uuid)

            # 결과 저장 (target, origin, uuid, train_path, lora_path, thumbnail_path)
            results.append([filename.split('.')[0], filename, file_uuid, output_dir, lora_path, ""])

            print(f"Processed: {filename} -> {new_filename}")

    # 결과를 CSV 파일로 저장
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['target', 'origin', 'uuid', 'train_path', 'lora_path', 'thumbnail_path'])
        csv_writer.writerows(results)

    print(f"\nProcessed {len(results)} images.")
    print(f"CSV saved to: {output_csv}")

    return output_csv, output_dir


def read_dataset_paths(csv_file):
    """Read dataset paths from CSV file
    # 0: target, 1: origin, 2: uuid, 3: train_path, 4: lora_path, 5: thumbnail_path
    using index 3, 5
    """
    arr = []
    with open(csv_file, 'r') as f:
        # read csv file f
        data = f.readlines()

        # 첫 번째 행이 헤더인 경우 제외
        if len(data) > 0 and 'target' in data[0].lower():
            data = data[1:]

        # row split by ',' and get 3rd index and 4th index
        data = [row.split(',') for row in data]
        for row in data:
            if len(row) >= 5:  # 최소 5개 열이 있는지 확인
                print(row[3], row[4])
                arr.append([row[3], row[4]])

    return arr


def create_config(image_dir):
    """Create config dictionary with the specified image_dir"""
    config = {
        "general": {
            "enable_bucket": True,  # 단순 로고는 bucket 불필요
            "shuffle_caption": False,  # 캡션 순서 고정
            "keep_tokens": 3

        },
        "datasets": [
            {
                "resolution": 512,  # 해상도 증가
                "batch_size": 3,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "class_tokens": "professional typography, high quality lettering, vector art, clean lines, precise curves, detailed typography, artistic font design",
                        # 더 구체적인 캡션
                        "num_repeats": 15,  # 적은 이미지 수 고려

                    }
                ]
            }
        ]
    }
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='이미지가 있는 디렉토리 경로')
    parser.add_argument('--base_command', type=str, default='python ./train_network.py', help='Base training command')
    parser.add_argument('--preprocess_only', action='store_true', help='이미지 전처리만 수행하고 학습은 건너뜁니다')
    args = parser.parse_args()

    # 이미지 세트 처리 (UUID 변환 및 텍스트 파일 생성)
    print(f"Processing images in: {args.image_dir}")
    csv_file, processed_dir = process_image_set(args.image_dir)

    # 전처리만 수행하는 경우 여기서 종료
    if args.preprocess_only:
        print(f"Preprocessing completed. CSV file created at: {csv_file}")
        return

    # read lora path from csv file
    dataset_paths = read_dataset_paths(csv_file)

    if not dataset_paths:
        print("No valid dataset paths found in CSV file.")
        return

    # Process each dataset path
    for i, obj in enumerate(dataset_paths, 1):
        dataset_path = obj[0]
        lora_path = obj[1].replace('\n', '')
        print(f"Processing dataset {i}/{len(dataset_paths)}")

        # 학습에 필요한 디렉토리 생성
        os.makedirs(lora_path, exist_ok=True)

        # Base command without config path
        base_cmd = (
            f'CUDA_VISIBLE_DEVICES=1 {args.base_command} '
            '--pretrained_model_name_or_path="/home/user/data/stable-diffusion-webui-forge/models/Stable-diffusion/Anything-v4.5-pruned.safetensors" '
            '--network_module=networks.lora '
            '--network_args "conv_dim=16" "conv_alpha=8" '  # 더 작게 감소
            '--network_dim=128 '  # 128에서 64로 감소
            '--network_alpha=64 '  # 64에서 32로 감소
            '--loss_type=smooth_l1 '  # 추가: Huber/smooth L1/MSE 손실 함수 선택
            '--huber_schedule=snr '  # 추가: 스케줄링 방법 선택
            '--huber_c=0.5 '  # 추가: Huber 손실 파라미터
            f'--output_dir="{lora_path}" '
            '--noise_offset=0.1 '
            '--optimizer_type=Lion '
            '--learning_rate=1e-5 '  # 5e-6에서 상향 - 더 적극적인 학습
            '--max_train_epochs=100 '  # 60에서 100으로 증가
            '--lr_scheduler=cosine_with_restarts '
            '--save_state_on_train_end '
            '--save_precision=fp16 '
            '--mixed_precision=fp16 '
            '--noise_offset_random_strength '
            '--gradient_accumulation_steps=4 '  # 그래디언트 누적으로 안정성 향상
            '--lr_scheduler_num_cycles=5 '  # 학습률 재시작 횟수 명시
            '--huber_schedule=snr'
        )

        print(f"\nProcessing dataset {os.path.basename(dataset_path)}/{len(dataset_paths)}")
        print(f"Dataset path: {dataset_path}")

        # Create config for this dataset
        # mkdir
        os.makedirs('./config', exist_ok=True)
        config = create_config(dataset_path)

        # Create config file path
        config_path = f"./config/config_{os.path.basename(dataset_path)}.toml"

        # Save config file
        with open(config_path, "w") as f:
            toml.dump(config, f)

        # Create training command for this dataset
        current_cmd = f'{base_cmd} --dataset_config="{config_path}" --output_name="{os.path.basename(dataset_path)}"'

        try:
            # Execute training command
            print(f"Training model for {dataset_path}")
            print(f"Using config file: {config_path}")
            subprocess.run(current_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing dataset {dataset_path}: {e}")
            continue

    send_google_chat_message("폰트 학습 완료")
    print("All training processes completed successfully.")

# 전처리 및 학습 모두 수행
# python danupapa-train.py --image_dir "경로/image-set"
# 전처리만 수행
#python danupapa-train.py --image_dir "경로/image-set" --preprocess_only
if __name__ == "__main__":
    main()