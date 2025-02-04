import csv

import pandas as pd
import os
import argparse
import subprocess
from pathlib import Path
import toml


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
            "enable_bucket": True
        },
        "datasets": [
            {
                "resolution": 512,
                "batch_size": 3,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "class_tokens": "clean_text",
                        "num_repeats": 10
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
            '--pretrained_model_name_or_path="./sd3/Anything-v4.5-pruned.safetensors" '
            '--network_module=networks.lora '
            '--network_dim=128 '
            '--network_alpha=64 '
            # '--save_every_n_epochs=5 '
            f'--output_dir="{lora_path}" '
            '--noise_offset=0.1 '
            '--optimizer_type=Lion '
            '--clip_skip=2 '
            '--learning_rate=5e-5 '
            '--max_train_epochs=80 '
            '--lr_scheduler=cosine_with_restarts '
            '--lr_warmup_steps=100 '
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


if __name__ == "__main__":
    main()