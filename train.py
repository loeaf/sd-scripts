import csv

import pandas as pd
import os
import argparse
import subprocess
from pathlib import Path
import json
import requests
import toml
GOOGLE_CHAT_WEBHOOK = "https://chat.googleapis.com/v1/spaces/AAAABLzMLsI/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=lwSnIq9XvRuT56A9BRh1xEuE-wU8vzqny_skSrTMIio"

def send_google_chat_message(message):
    """Google Chat으로 메시지를 보내는 함수"""
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {'text': message}
    response = requests.post(GOOGLE_CHAT_WEBHOOK, headers=headers, data=json.dumps(data))
    return response.status_code


def read_dataset_paths(csv_file):
    """Read dataset paths from CSV file
    # 0: target, 1: origin, 2: uuid, 3: train_path, 4: lora_path, 5: sumnail_path
    using index 3, 5
    """
    arr = []
    with open(csv_file, 'r') as f:
        # read csv file f
        data = f.readlines()

        # row split by ',' and get 3rd index and 5th index
        data = [row.split(',') for row in data]
        for row in data:
            print(row[3], row[4])
            arr.append([row[3], row[4]])

    return arr



def create_config(image_dir):
    """Create config dictionary with the specified image_dir"""
    config = {
        "general": {
            "enable_bucket": False,  # 단순 로고는 bucket 불필요
            "shuffle_caption": False  # 캡션 순서 고정
        },
        "datasets": [
            {
                "resolution": 512,
                "batch_size": 3,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "class_tokens": "sandoll style, letter, black and white",  # 더 구체적인 캡션
                        "num_repeats": 10  # 적은 이미지 수 고려
                    }
                ]
            }
        ]
    }
    return config

# python train.py --csv_file /home/user/data/resource/font_pairs.csv
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='./font_pairs.csv', help='Path to CSV file containing dataset paths')
    parser.add_argument('--base_command', type=str, default='python ./train_network.py', help='Base training command')
    args = parser.parse_args()

    # read lora path from csv file
    dataset_paths = read_dataset_paths(args.csv_file)

    # Process each dataset path
    for i, obj in enumerate(dataset_paths, 1):
        dataset_path = obj[0]
        lora_path = obj[1].replace('\n', '')
        print(f"Processing dataset {i}/{len(dataset_paths)}")
        # Base command without config path
        base_cmd = (
            f'CUDA_VISIBLE_DEVICES=1 {args.base_command} '
            '--pretrained_model_name_or_path="/home/user/data/stable-diffusion-webui-forge/models/Stable-diffusion/sd_2-1.safetensors" '
            '--network_module=networks.lora '
            '--network_dim=64 '
            '--network_alpha=32 '
            # '--save_every_n_epochs=5 '
            f'--output_dir="{lora_path}" '
            '--noise_offset=0.1 '
            '--optimizer_type=Lion '
            '--clip_skip=2 '
            '--learning_rate=1e-5 '
            '--max_train_epochs=200 '
            '--lr_scheduler=cosine_with_restarts '
            # '--lr_warmup_steps=100 '
            '--save_state_on_train_end '
            '--save_precision=fp16 '
            '--mixed_precision=fp16 '
            '--noise_offset_random_strength '
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
        # finally:
            # Clean up config file
            # if os.path.exists(config_path):
            #     os.remove(config_path)
    send_google_chat_message("폰트 학습 완료")


if __name__ == "__main__":
    main()