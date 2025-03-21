import csv
import os
import argparse
import subprocess
from pathlib import Path
import json
import requests
import toml
import uuid
import shutil
import datetime

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
    모든 이미지가 하나의 학습 세트로 사용됩니다.

    Args:
        image_dir (str): 이미지 파일들이 있는 디렉토리 경로

    Returns:
        tuple: (처리된 이미지 디렉토리 경로, LoRA 모델 저장 경로)
    """
    # 이미지 확장자 목록
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    processed_images = []

    # 타임스탬프로 고유한 폴더명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = Path(image_dir).name

    # 결과를 저장할 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(image_dir), f"processed_{dir_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # LoRA 저장 경로 생성 (단일 LoRA 모델을 위한 경로)
    lora_dir = os.path.join(os.path.dirname(image_dir), f"lora_{dir_name}_{timestamp}")
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

            # 동일한 이름의 텍스트 파일 생성
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f"danupapa, sd character, single character")

            processed_images.append(new_filename)
            print(f"Processed: {filename} -> {new_filename}")

    print(f"\nProcessed {len(processed_images)} images.")
    print(f"Images saved to: {output_dir}")
    print(f"LoRA will be saved to: {lora_dir}")

    return output_dir, lora_dir

# find /data/train/lora_real-image-set_20250321_130747/ -name "*.safetensors" -exec mv {} /home/user/data/stable-diffusion-webui-forge/models/Lora \;
# python danupapa-train.py --image_dir "/data/train/real-image-set"

def create_config(image_dir):
    """Create config dictionary with the specified image_dir"""
    config = {
        "general": {
            "enable_bucket": True,
            "shuffle_caption": False,
            "keep_tokens": 3
        },
        "datasets": [
            {
                "resolution": 512,
                "batch_size": 2,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "class_tokens": "professional typography, high quality lettering, vector art, clean lines, precise curves, detailed typography, artistic font design",
                        "num_repeats": 15,
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
    processed_dir, lora_dir = process_image_set(args.image_dir)

    # 전처리만 수행하는 경우 여기서 종료
    if args.preprocess_only:
        print(f"Preprocessing completed. Processed images are in: {processed_dir}")
        return

    # 학습 설정 생성
    print(f"Creating training configuration...")

    # Create config for this dataset
    os.makedirs('./config', exist_ok=True)
    config = create_config(processed_dir)

    # Create config file path
    config_name = os.path.basename(processed_dir)
    config_path = f"./config/config_{config_name}.toml"

    # Save config file
    with open(config_path, "w") as f:
        toml.dump(config, f)
    # 수정된 명령어
    base_cmd = (
        f'CUDA_VISIBLE_DEVICES=1 {args.base_command} '
        '--pretrained_model_name_or_path="/home/user/data/stable-diffusion-webui-forge/models/Stable-diffusion/celestial.safetensors" '
        '--network_module=networks.lora '
        '--network_args "conv_dim=16" "conv_alpha=8" '
        '--network_dim=64 '  # 128에서 64로 줄임
        '--network_alpha=32 '  # 64에서 32로 줄임
        '--loss_type=smooth_l1 '
        '--huber_schedule=snr '
        '--huber_c=0.5 '
        f'--output_dir="{lora_dir}" '
        '--noise_offset=0.1 '
        '--optimizer_type=Lion '
        '--learning_rate=1e-5 '
        '--max_train_epochs=15 '  # 100에서 15로 크게 줄임
        '--lr_scheduler=cosine_with_restarts '
        '--save_state_on_train_end '
        '--save_precision=fp16 '
        '--mixed_precision=fp16 '
        '--noise_offset_random_strength '
        '--gradient_accumulation_steps=4 '
        '--lr_scheduler_num_cycles=2 '  # 5에서 2로 줄임
        '--clip_skip=2 '  # 추가: 더 일반적인 특징 캡처
        '--save_every_n_epochs=1 '  # 추가: 중간 체크포인트 저장
        '--huber_schedule=snr'
    )

    # Create training command for this dataset
    output_name = os.path.basename(args.image_dir)
    current_cmd = f'{base_cmd} --dataset_config="{config_path}" --output_name="{output_name}"'

    try:
        # Execute training command
        print(f"Training model with images from: {processed_dir}")
        print(f"Using config file: {config_path}")
        print(f"LoRA model will be saved to: {lora_dir}")
        subprocess.run(current_cmd, shell=True, check=True)
        send_google_chat_message(f"폰트 학습 완료: {output_name}")
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        send_google_chat_message(f"폰트 학습 오류 발생: {output_name} - {str(e)}")


if __name__ == "__main__":
    main()