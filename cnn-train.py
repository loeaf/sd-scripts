import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import random
import json
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob
import shutil
from collections import Counter
# 1. 더 강력한 정규화와 함께 더 깊은 모델 사용
from torchvision.models import resnet50, ResNet50_Weights

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import albumentations as al
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import random
import json
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob
import shutil
from collections import Counter


def resize_with_padding(image, target_size=(224, 224)):
    # 원본 이미지 크기
    h, w = image.shape[:2]

    # 대상 크기
    target_h, target_w = target_size

    # 비율 계산 (가로, 세로 중 작은 비율 선택)
    ratio = min(target_w / w, target_h / h)

    # 새 크기 계산
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # 이미지 리사이징
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 새 캔버스 생성 (흰색 배경)
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    # 이미지를 캔버스 중앙에 배치
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # 리사이징된 이미지를 캔버스에 복사
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas
# 클래스 균형을 맞추는 샘플러 생성 함수
def create_balanced_sampler(labels):
    """클래스 균형을 맞추는 샘플러 생성"""
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]

    # 가중치를 torch 텐서로 변환
    weights = torch.from_numpy(weights).float()

    # 클래스 균형을 맞추는 WeightedRandomSampler 생성
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler
# 먼저 CSV 파일에서 filtername 추출 및 클래스 정의 함수
def extract_filternames_from_csv(csv_path):
    """CSV 파일에서 고유한 filtername을 추출하고 클래스 매핑을 생성합니다."""
    print(f"CSV 파일에서 filtername 분석 중: {csv_path}")

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # filtername 컬럼 추출
    if 'filtername' not in df.columns:
        raise ValueError(f"CSV 파일에 'filtername' 컬럼이 없습니다: {csv_path}")

    # 모든 필터네임 수집
    all_filternames = []
    for idx, row in df.iterrows():
        if pd.isna(row['filtername']) or not isinstance(row['filtername'], str):
            print(f"Warning: 행 {idx}에 유효하지 않은 filtername이 있습니다.")
            continue

        # 쉼표로 분리하고 공백 제거
        names = [name.strip() for name in row['filtername'].split(',')]
        all_filternames.extend(names)

    # 고유한 필터네임 찾기
    unique_filternames = sorted(set(filter(None, all_filternames)))

    print(f"총 {len(unique_filternames)}개의 고유한 filtername을 찾았습니다.")

    # 클래스 매핑 생성
    filtername_to_label = {name: idx for idx, name in enumerate(unique_filternames)}

    # 각 클래스 당 샘플 수 계산
    class_counts = Counter(all_filternames)

    # 클래스 정보를 파일로 저장 (디버깅용)
    with open('filter_classes.txt', 'w', encoding='utf-8') as f:
        f.write(f"총 클래스 수: {len(unique_filternames)}\n\n")
        f.write("클래스 ID, 필터네임, 샘플 수\n")
        for name, idx in filtername_to_label.items():
            f.write(f"{idx}, {name}, {class_counts[name]}\n")

    print(f"클래스 정보가 'filter_classes.txt'에 저장되었습니다.")

    return unique_filternames, filtername_to_label, class_counts


# Function for multiprocessing - needs to be at module level
def create_unicode_font_image(params):
    """Generate an image with 1-4 Unicode characters using the specified font"""
    chars, font_path, font_id, filternames, char_index, image_size, output_dir = params

    # chars는 이제 1~4글자의 문자열
    text = chars  # 여러 글자로 구성된 텍스트

    # 결과를 저장할 리스트
    results = []

    # 각 필터네임에 대해 이미지 생성
    for filtername in filternames:
        # 필터네임을 폴더 이름으로 안전하게 변환
        safe_filtername = filtername.replace('/', '_').replace(' ', '_')

        # 파일명을 안전하게 만들기 위해 텍스트의 유니코드 코드 포인트 사용
        text_code = "_".join([f"U{ord(c):04X}" for c in text])
        out_dir = os.path.join(output_dir, safe_filtername)
        os.makedirs(out_dir, exist_ok=True)
        image_path = os.path.join(out_dir, f"{font_id}_{text_code}_{char_index}.png")

        # Skip if file already exists (for resuming interrupted processing)
        if os.path.exists(image_path):
            # Verify the file is valid
            try:
                with Image.open(image_path) as img:
                    # Just accessing a property to verify image is valid
                    img_format = img.format
                results.append((image_path, filtername))
                continue
            except Exception:
                # Remove corrupted file
                try:
                    os.remove(image_path)
                except:
                    pass

        # Check if font file exists
        if not os.path.exists(font_path):
            continue

        image = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(image)

        # 이미지 생성 및 글자 그리기 부분만 수정
        try:
            # 텍스트 크기에 맞게 폰트 크기 조정
            font_size = int(min(image_size) * 0.7)
            font = ImageFont.truetype(font_path, font_size)

            # 폰트 크기 자동 조정 (텍스트 길이에 따라)
            while True:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 텍스트가 이미지 크기의 80%를 넘지 않도록 조정
                if text_width > image_size[0] * 0.8 or text_height > image_size[1] * 0.8:
                    font_size -= 1
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    break

                # 너무 작아지지 않도록 제한
                if font_size < 10:
                    break

            # 텍스트 중앙 정렬
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) // 2 - bbox[0],
                        (image_size[1] - text_height) // 2 - bbox[1])

            # 텍스트 그리기
            draw.text(position, text, font=font, fill='black')

            # 이미지 저장
            image.save(image_path)

            # 비율을 유지하면서 리사이징
            with Image.open(image_path) as img:
                # 원본 이미지의 크기
                width, height = img.size

                # 비율 계산 (가로, 세로 중 작은 비율 선택)
                ratio = min(224 / width, 224 / height)

                # 새 크기 계산
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                # 비율을 유지하면서 리사이징
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                # 새 캔버스 생성 (흰색 배경)
                new_img = Image.new('RGB', (224, 224), color='white')

                # 이미지를 캔버스 중앙에 배치
                paste_x = (224 - new_width) // 2
                paste_y = (224 - new_height) // 2
                new_img.paste(resized_img, (paste_x, paste_y))

                # 리사이징된 이미지 저장
                new_img.save(image_path)

            # Verify the image was saved correctly
            try:
                with Image.open(image_path) as img:
                    img_format = img.format
                results.append((image_path, filtername))
            except Exception as e:
                # Remove corrupted file and print error
                print(f"이미지 검증 오류: {e}")
                try:
                    os.remove(image_path)
                except:
                    pass

        except Exception as e:
            print(f"이미지 생성 오류: {e}")
            pass

    return results

# Font Image Generator
@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (224, 224)  # 기존 128x128에서 224x224로 변경
    output_dir: str = "datasets/filter_dataset"
    korean_unicode_file: str = "union_korean_unicodes.json"
    num_samples_per_font: int = 20  # Samples per font
    num_processes: int = None  # Will default to number of CPU cores
    clean_output_dir: bool = False  # Set to True to remove existing dataset

    def __post_init__(self):
        # Optionally clean output directory
        if self.clean_output_dir and os.path.exists(self.output_dir):
            print(f"Cleaning output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        # Set number of processes if not specified
        if self.num_processes is None:
            self.num_processes = cpu_count()

        # Load Korean Unicode characters from JSON file
        try:
            with open(self.korean_unicode_file, 'r') as f:
                self.korean_unicodes = json.load(f)
            print(f"Loaded {len(self.korean_unicodes)} Korean Unicode characters from {self.korean_unicode_file}")
        except Exception as e:
            print(f"Failed to load Korean Unicode file: {e}")
            # Fallback to a small set of common Hangul
            self.korean_unicodes = list(range(44032, 44032 + 100))  # First 100 Hangul syllables
            print(f"Using fallback set of {len(self.korean_unicodes)} Korean Unicode characters")

        # Convert Unicode code points to actual characters
        self.korean_chars = [chr(code) for code in self.korean_unicodes]

    def verify_dataset(self):
        """Verify all images in the dataset are valid, removing corrupted ones"""
        print("데이터셋 무결성 검증 중...")
        count_before = len(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True))

        corrupted = 0
        for img_path in tqdm(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True),
                             desc="이미지 확인 중"):
            try:
                with Image.open(img_path) as img:
                    # Just accessing a property to verify image
                    img_format = img.format
            except Exception:
                # Remove corrupted file
                try:
                    os.remove(img_path)
                    corrupted += 1
                except:
                    pass

        count_after = len(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True))
        print(f"데이터셋 검증: {corrupted}개의 손상된 이미지를 발견하고 제거했습니다.")
        print(f"데이터셋 크기: {count_before} -> {count_after} 이미지")

    def generate_dataset_from_csv(self, csv_path: str):
        """Generate a dataset from the font CSV file using multiprocessing"""
        start_time = time.time()

        # CSV 파일에서 필터네임 추출 및 클래스 매핑 생성
        unique_filternames, filtername_to_label, class_counts = extract_filternames_from_csv(csv_path)

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Filter out rows where file doesn't exist or filtername is invalid
        valid_rows = []
        for _, row in df.iterrows():
            if not os.path.exists(row['FilePath']):
                print(f"Warning: Font file not found: {row['FilePath']}")
                continue

            if pd.isna(row['filtername']) or not isinstance(row['filtername'], str):
                print(f"Warning: Invalid filtername for font_id {row['font_id']}")
                continue

            valid_rows.append(row)

        if len(valid_rows) == 0:
            print("Error: No valid font files found!")
            return [], [], {}

        df_valid = pd.DataFrame(valid_rows)
        print(f"Found {len(df_valid)} valid font files out of {len(df)} total")

        # Prepare tasks for parallel processing
        tasks = []

        for _, row in df_valid.iterrows():
            font_id = row['font_id']
            font_path = row['FilePath']

            # 필터네임을 쉼표로 분리
            if pd.isna(row['filtername']) or not isinstance(row['filtername'], str):
                continue

            filternames = [name.strip() for name in row['filtername'].split(',') if name.strip()]

            # 유효한 필터네임이 없으면 건너뛰기
            if not filternames:
                continue

            # 1글자, 2글자, 3글자, 4글자 텍스트 조합 생성
            texts_to_use = []

            # 1글자 (기존과 동일)
            if len(self.korean_chars) <= self.num_samples_per_font // 4:
                single_chars = self.korean_chars
            else:
                random.seed(42)
                single_chars = random.sample(self.korean_chars, self.num_samples_per_font // 4)

            texts_to_use.extend([c for c in single_chars])

            # 2글자 조합
            for _ in range(self.num_samples_per_font // 4):
                random.seed(42 + _)  # 다양한 조합을 위해 시드 변경
                chars = random.sample(self.korean_chars, 2)
                texts_to_use.append(''.join(chars))

            # 3글자 조합
            for _ in range(self.num_samples_per_font // 4):
                random.seed(142 + _)
                chars = random.sample(self.korean_chars, 3)
                texts_to_use.append(''.join(chars))

            # 4글자 조합
            for _ in range(self.num_samples_per_font // 4):
                random.seed(242 + _)
                chars = random.sample(self.korean_chars, 4)
                texts_to_use.append(''.join(chars))

            # 각 텍스트에 대해 이미지 생성 태스크 추가
            for i, text in enumerate(texts_to_use):
                tasks.append((text, font_path, font_id, filternames, i, self.image_size, self.output_dir))

        # Generate images in parallel using multiprocessing
        print(f"Generating images for {len(tasks)} tasks using {self.num_processes} processes...")

        image_paths = []
        labels = []

        # Use multiprocessing pool with chunking for better performance
        with Pool(processes=self.num_processes - 5) as pool:
            # Use imap_unordered with chunking for better performance
            chunksize = max(1, len(tasks) // (self.num_processes * 10))
            all_results = list(tqdm(
                pool.imap_unordered(create_unicode_font_image, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="이미지 생성 중"
            ))

            # Process results - each result is a list of tuples (img_path, filtername)
            for results in all_results:
                for img_path, filtername in results:
                    if img_path and filtername in filtername_to_label:
                        image_paths.append(img_path)
                        labels.append(filtername_to_label[filtername])

        # Verify dataset integrity
        self.verify_dataset()

        # Check final class distribution
        label_counts = Counter(labels)
        print("\n최종 클래스 분포:")
        for name, idx in filtername_to_label.items():
            count = label_counts.get(idx, 0)
            print(f"  {name}: {count} 이미지")

        # Save the final distribution to file
        with open('final_class_distribution.txt', 'w', encoding='utf-8') as f:
            f.write("클래스 ID, 필터네임, 생성된 이미지 수\n")
            for name, idx in filtername_to_label.items():
                count = label_counts.get(idx, 0)
                f.write(f"{idx}, {name}, {count}\n")

        print(f"최종 클래스 분포가 'final_class_distribution.txt'에 저장되었습니다.")

        end_time = time.time()
        print(
            f"Generated {len(image_paths)} images for {len(filtername_to_label)} filter types in {end_time - start_time:.2f} seconds")
        return image_paths, labels, filtername_to_label


# Albumentations 증강 함수
def get_train_transforms(use_gray=False):
    """Albumentations 라이브러리를 사용한 강력한 데이터 증강"""
    return al.Compose([
        al.OneOf([
            al.Rotate(limit=(-35, 35), border_mode=cv2.BORDER_CONSTANT),
        ], p=0.05),
        al.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=30, p=0.05),
        al.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, distort_limit=5.0, shift_limit=0.1, p=0.05),
        al.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.05),
        al.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=15, p=0.05),

        al.RandomGridShuffle(p=0.05),

        al.RandomGamma(p=0.05),
        al.HueSaturationValue(p=0.05),
        al.RGBShift(p=0.05),
        al.CLAHE(p=0.05),
        al.ChannelShuffle(p=0.05),
        al.InvertImg(p=0.05),

        al.RandomSnow(p=0.05),
        al.RandomRain(p=0.05),
        al.RandomSunFlare(p=0.05, num_flare_circles_lower=1, num_flare_circles_upper=2, src_radius=110),
        # al.RandomShadow(p=0.05),
        al.RandomBrightnessContrast(p=0.05),
        al.GaussNoise(p=0.05),
        al.ISONoise(p=0.05),
        al.MultiplicativeNoise(p=0.05),
        al.ToGray(p=1.0 if use_gray else 0.05),
        al.ToSepia(p=0.05),
        al.Solarize(p=0.05),
        al.Equalize(p=0.05),
        al.Posterize(p=0.05),
        al.FancyPCA(p=0.05),
        al.OneOf([
            al.MotionBlur(blur_limit=3),
            al.Blur(blur_limit=3),
            al.MedianBlur(blur_limit=3),
            al.GaussianBlur(blur_limit=3),
        ], p=0.05),
        al.CoarseDropout(p=0.05),
        al.Cutout(p=0.05),
        al.GridDropout(p=0.05),
        al.ChannelDropout(p=0.05),
        al.Downscale(p=0.1),
        al.ImageCompression(quality_lower=60, p=0.1),
        al.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    """검증 및 테스트 데이터에 대한 기본 변환"""
    return al.Compose([
        ToTensorV2(),
    ])


# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Reshape query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C//8
        proj_key = self.key(x).view(batch_size, -1, width * height)  # B x C//8 x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # B x (W*H) x (W*H)
        attention = self.softmax(energy)  # B x (W*H) x (W*H)

        proj_value = self.value(x).view(batch_size, -1, width * height)  # B x C x (W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (W*H)
        out = out.view(batch_size, C, width, height)  # B x C x W x H

        out = self.gamma * out + x
        return out


# Channel Attention Module (Squeeze and Excitation)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


# CBAM (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# AttentionFilterClassifier 모델 수정
class AttentionFilterClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AttentionFilterClassifier, self).__init__()

        # 더 깊은 네트워크 구조로 변경
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cbam1 = CBAM(64)

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cbam2 = CBAM(128)

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.self_attention = SelfAttention(256)  # Self-attention after third block

        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),  # 추가
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cbam4 = CBAM(512)

        # Global pooling and classification
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Apply convolution blocks with attention
        x = self.conv1(x)
        x = self.cbam1(x)

        x = self.conv2(x)
        x = self.cbam2(x)

        x = self.conv3(x)
        x = self.self_attention(x)

        x = self.conv4(x)
        x = self.cbam4(x)

        # Global pooling and classification
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# On-the-fly Dataset class with Albumentations support
class AlbumentationsDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        # Filter out any paths that don't exist or are corrupted
        valid_items = []
        for path, label in zip(image_paths, labels):
            try:
                if os.path.exists(path):
                    # Try to open to verify it's a valid image
                    with Image.open(path) as img:
                        img_format = img.format  # Just to verify it's readable
                    valid_items.append((path, label))
            except Exception:
                pass

        # Unpack valid items
        self.image_paths, self.labels = zip(*valid_items) if valid_items else ([], [])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    # AlbumentationsDataset 클래스에서 오류 발생 시 이미지 크기 수정
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # 이미지 읽기
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 비율을 유지하면서 리사이징
            image = resize_with_padding(image, target_size=(224, 224))

            label = self.labels[idx]

            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]

            return image, label
        except Exception as e:
            print(f"이미지 로딩 오류 {image_path}: {e}")
            # 오류 시 빈 이미지 생성
            image = np.ones((224, 224, 3), dtype=np.uint8) * 255  # 흰색 배경
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            return image, self.labels[idx]


# CNN Model Architecture for filtername classification
class FilterClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FilterClassifierCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    # For tracking metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap with try-except to handle any unexpected errors during training
        try:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            epoch_time = time.time() - epoch_start

            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Time: {epoch_time:.2f}s')

            # 100 에포크마다 모델 저장
            if (epoch + 1) % 100 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'history': history
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
                print(f'Checkpoint saved for epoch {epoch + 1}')

            # 최고 성능 모델 따로 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                early_stopping_counter = 0

                # 최고 성능 모델 저장
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'history': history
                }
                torch.save(best_checkpoint, 'best_filter_classifier.pth')
                print(f'Best model saved with Val Acc: {val_acc:.2f}%')
            else:
                early_stopping_counter += 1
                print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')

            # Early stopping
            if early_stopping_counter >= patience:
                print(
                    f'Early stopping triggered after epoch {epoch + 1}. Best epoch was {best_epoch + 1} with validation accuracy: {best_val_acc:.2f}%')
                break

        except Exception as e:
            print(f"Error during training epoch {epoch + 1}: {e}")
            # Skip to next epoch
            continue

    print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}')
    return model, history


# Model evaluation function
def evaluate_model(model, test_loader, device='cuda', label_map=None):
    model.eval()
    correct = 0
    total = 0

    # For confusion matrix
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Print per-class accuracy
    if label_map:
        idx_to_name = {idx: name for name, idx in label_map.items()}
        class_correct = {}
        class_total = {}

        for pred, label in zip(all_predictions, all_labels):
            label_name = idx_to_name[label]
            if label_name not in class_correct:
                class_correct[label_name] = 0
                class_total[label_name] = 0

            class_total[label_name] += 1
            if pred == label:
                class_correct[label_name] += 1

        print("\n클래스별 정확도:")
        for name in sorted(class_correct.keys()):
            acc = 100 * class_correct[name] / max(1, class_total[name])
            print(f"{name}: {acc:.2f}% ({class_correct[name]}/{class_total[name]})")

        # Save detailed evaluation results
        with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"전체 테스트 정확도: {accuracy:.2f}%\n\n")
            f.write("클래스별 정확도:\n")
            for name in sorted(class_correct.keys()):
                acc = 100 * class_correct[name] / max(1, class_total[name])
                f.write(f"{name}: {acc:.2f}% ({class_correct[name]}/{class_total[name]})\n")

    return accuracy, all_predictions, all_labels


# 구조가 간소화된 Attention 모델
class SimpleAttentionFilterClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cbam1 = CBAM(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.self_attention = SelfAttention(128)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)
        x = self.conv2(x)
        x = self.self_attention(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 사전 학습된 ResNet 기반 모델
class PretrainedModelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 더 깊은 ResNet 사용
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)

        # 특성 추출기 부분 고정 (선택 사항)
        # for param in list(self.resnet.parameters())[:-30]:
        #     param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.7),  # 더 강력한 드롭아웃
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
# On-the-fly Dataset class with error handling
class FontDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # Filter out any paths that don't exist or are corrupted
        valid_items = []
        for path, label in zip(image_paths, labels):
            try:
                if os.path.exists(path):
                    # Try to open to verify it's a valid image
                    with Image.open(path) as img:
                        img_format = img.format  # Just to verify it's readable
                    valid_items.append((path, label))
            except Exception:
                pass

        # Unpack valid items
        self.image_paths, self.labels = zip(*valid_items) if valid_items else ([], [])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            # If there's still an error, return a black image with the same label
            # This is a last resort fallback
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (128, 128), color='black')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]


# Main execution function
def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Initialize the generator and create dataset with larger images
    generator = FontImageGenerator(
        korean_unicode_file="union_korean_unicodes.json",
        num_samples_per_font=20,
        num_processes=cpu_count(),
        clean_output_dir=False,  # 새 이미지 크기로 다시 생성하려면 True로 설정
        image_size=(224, 224)  # 큰 이미지 크기 지정
    )

    csv_path = 'cnn-cate_filter_merged.csv'

    print(f"Generating dataset from fonts using {generator.num_processes} processes...")
    image_paths, labels, label_map = generator.generate_dataset_from_csv(csv_path)

    if not image_paths:
        print("Error: No valid images were generated. Please check font paths and permissions.")
        return

    # Split the dataset
    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    except ValueError as e:
        print(f"Error in train/test split: {e}")
        print("Using simple split without stratification")
        total = len(image_paths)
        train_idx = int(0.7 * total)
        val_idx = int(0.85 * total)

        train_paths, train_labels = image_paths[:train_idx], labels[:train_idx]
        val_paths, val_labels = image_paths[train_idx:val_idx], labels[train_idx:val_idx]
        test_paths, test_labels = image_paths[val_idx:], labels[val_idx:]

    # Set up Albumentations transforms
    train_transforms = get_train_transforms(use_gray=False)
    val_transforms = get_val_transforms()

    # Create datasets and dataloaders with Albumentations
    train_dataset = AlbumentationsDataset(train_paths, train_labels, transforms=train_transforms)
    val_dataset = AlbumentationsDataset(val_paths, val_labels, transforms=val_transforms)
    test_dataset = AlbumentationsDataset(test_paths, test_labels, transforms=val_transforms)

    # Check if datasets are valid
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Empty dataset after filtering invalid images.")
        return

    # 균형 있는 샘플링을 위한 sampler 생성
    sampler = create_balanced_sampler(train_labels)

    # DataLoader 생성 시 sampler 사용 (shuffle=True는 제거)
    train_loader = DataLoader(
        train_dataset,
        batch_size=486,
        sampler=sampler,  # shuffle=True 대신 sampler 사용
        num_workers=8,
        pin_memory=True,
    multiprocessing_context='spawn'  # 추가: 'fork' 대신 'spawn' 사용
    )

    val_loader = DataLoader(val_dataset, batch_size=486, shuffle=False, num_workers=8, pin_memory=True,
    multiprocessing_context='spawn')
    test_loader = DataLoader(test_dataset, batch_size=486, shuffle=False, num_workers=8, pin_memory=True,
    multiprocessing_context='spawn')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # main 함수 내에서 모델 초기화 부분 수정
    num_classes = len(label_map)

    # 모델 유형 선택 (simple_attention, pretrained_resnet)
    model_type = 'pretrained_resnet'  # 또는 'simple_attention'

    if model_type == 'simple_attention':
        model = SimpleAttentionFilterClassifier(num_classes)
        print(f"Using Simplified Attention CNN with {num_classes} classes")
    else:
        model = PretrainedModelClassifier(num_classes)
        print(f"Using Pretrained ResNet18 with {num_classes} classes")

    # 다중 GPU 사용을 위한 코드 추가
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)


    # 모델을 device로 옮기기
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    print("Starting training...")
    try:
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=5000,
            patience=20,
            device=device
        )

        print("Training completed!")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        # Reverse the label map for evaluation
        reverse_label_map = {idx: name for name, idx in label_map.items()}
        test_acc, all_preds, all_labels = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            label_map=reverse_label_map
        )

        # Save the final model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'label_map': label_map,
            'history': history,
            'test_accuracy': test_acc
        }, 'filter_classifier_final.pth')

        print("Model saved to filter_classifier_final.pth")

    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save the model anyway in case of partial training
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_map': label_map,
        }, 'filter_classifier_partial.pth')
        print("Partial model saved to filter_classifier_partial.pth")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    main()