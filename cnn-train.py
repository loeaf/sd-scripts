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


# 라벨 값 저장 함수
def save_label_map(label_map, output_dir="label_maps"):
    """라벨 맵을 여러 형식으로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    # 역방향 라벨 맵 생성 (인덱스 -> 필터네임)
    reverse_label_map = {idx: name for name, idx in label_map.items()}

    # 1. JSON 형식으로 저장
    label_data = {
        "filtername_to_label": label_map,
        "label_to_filtername": reverse_label_map
    }

    json_path = os.path.join(output_dir, "label_map.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

    # 2. CSV 형식으로 저장
    csv_data = []
    for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
        csv_data.append([idx, name])

    csv_path = os.path.join(output_dir, "label_map.csv")
    pd.DataFrame(csv_data, columns=['label_id', 'filter_name']).to_csv(
        csv_path, index=False, encoding='utf-8'
    )

    # 3. Python 모듈로 저장
    py_path = os.path.join(output_dir, "label_map.py")
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write("# 필터네임 -> 인덱스 매핑\n")
        f.write("FILTER_TO_INDEX = {\n")
        for name, idx in sorted(label_map.items()):
            f.write(f"    '{name}': {idx},\n")
        f.write("}\n\n")

        f.write("# 인덱스 -> 필터네임 매핑\n")
        f.write("INDEX_TO_FILTER = {\n")
        for idx, name in sorted(reverse_label_map.items()):
            f.write(f"    {idx}: '{name}',\n")
        f.write("}\n")

    # 4. 텍스트 파일로 저장 (읽기 쉬운 형식)
    txt_path = os.path.join(output_dir, "label_map.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"총 클래스 수: {len(label_map)}\n\n")
        f.write("ID | 필터 이름\n")
        f.write("-" * 40 + "\n")
        for idx, name in sorted(reverse_label_map.items()):
            f.write(f"{idx:2d} | {name}\n")

    print(f"라벨 맵이 다음 파일로 저장되었습니다:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Python: {py_path}")
    print(f"  - Text: {txt_path}")


class DynamicFontDataset(Dataset):
    def __init__(self, font_data, label_map, transforms=None, target_size=(224, 224), samples_per_font=200):
        self.font_data = font_data
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.transforms = transforms
        self.target_size = target_size
        self.samples_per_font = samples_per_font
        self.korean_chars = self.load_korean_chars()
    # 폰트개수
    def __getitem__(self, idx):
        font_idx = idx // self.samples_per_font
        sample_idx = idx % self.samples_per_font
        font_path, font_id, filternames = self.font_data[font_idx]

        # 다중 레이블 벡터 생성
        label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for filtername in filternames:
            if filtername in self.label_map:
                label_vector[self.label_map[filtername]] = 1.0

        text_length = (sample_idx % 4) + 1
        random.seed(font_idx * 100 + sample_idx)
        text = ''.join(random.sample(self.korean_chars, text_length))

        try:
            image = self.create_text_image(text, font_path)
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return image, label_vector
        except Exception as e:
            print(f"이미지 생성 오류 (폰트: {font_id}, 텍스트: {text}): {e}")
            image = torch.zeros((3, *self.target_size), dtype=torch.float32)
            return image, label_vector

    def load_korean_chars(self):
        try:
            with open("union_korean_unicodes.json", 'r') as f:
                korean_unicodes = json.load(f)
            return [chr(code) for code in korean_unicodes]
        except:
            # 기본 한글 집합 (초성 + 중성 조합의 첫 100개)
            return [chr(code) for code in range(44032, 44032 + 100)]

    def __len__(self):
        # 폰트 수 * 텍스트 샘플 수
        return len(self.font_data) * self.samples_per_font  # 폰트당 20개 샘플 생성

    def create_text_image(self, text, font_path):
        """텍스트 이미지 생성"""
        image = np.ones((self.target_size[1], self.target_size[0], 3), dtype=np.uint8) * 255

        try:
            # PIL 이미지로 변환
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            # 글꼴 로드 및 크기 조정
            font_size = int(min(self.target_size) * 0.7)
            font = ImageFont.truetype(font_path, font_size)

            # 텍스트 크기 조정
            while True:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width > self.target_size[0] * 0.8 or text_height > self.target_size[1] * 0.8:
                    font_size -= 1
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    break

                if font_size < 10:
                    break

            # 텍스트 중앙 배치
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((self.target_size[0] - text_width) // 2 - bbox[0],
                        (self.target_size[1] - text_height) // 2 - bbox[1])

            # 텍스트 그리기
            draw.text(position, text, font=font, fill='black')

            # numpy 배열로 변환
            return np.array(pil_image)

        except Exception as e:
            print(f"텍스트 이미지 생성 오류: {e}")
            return image


def prepare_dynamic_dataset(csv_path):
    """CSV에서 폰트 데이터 로드 및 동적 데이터셋 준비"""
    print(f"CSV 파일에서 폰트 데이터 로드 중: {csv_path}")

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 유효한 필터네임 추출 및 클래스 매핑 생성
    all_filternames = []
    for idx, row in df.iterrows():
        if pd.isna(row['filtername']) or not isinstance(row['filtername'], str):
            continue
        names = [name.strip() for name in row['filtername'].split(',')]
        all_filternames.extend(names)

    unique_filternames = sorted(set(filter(None, all_filternames)))
    filtername_to_label = {name: idx for idx, name in enumerate(unique_filternames)}

    # 폰트 데이터 준비
    font_data = []
    for _, row in df.iterrows():
        if not os.path.exists(row['FilePath']):
            continue

        if pd.isna(row['filtername']) or not isinstance(row['filtername'], str):
            continue

        filternames = [name.strip() for name in row['filtername'].split(',') if name.strip()]
        if not filternames:
            continue

        font_data.append((row['FilePath'], row['font_id'], filternames))

    print(f"총 {len(font_data)}개의 유효한 폰트 파일과 {len(unique_filternames)}개의 필터 클래스를 찾았습니다.")

    return font_data, filtername_to_label

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

def create_unicode_font_image(params):
    chars, font_path, font_id, filternames, char_index, image_size, output_dir = params
    text = chars
    results = []

    for filtername in filternames:
        safe_filtername = filtername.replace('/', '_').replace(' ', '_')
        text_code = "_".join([f"U{ord(c):04X}" for c in text])
        out_dir = os.path.join(output_dir, safe_filtername)
        os.makedirs(out_dir, exist_ok=True)
        image_path = os.path.join(out_dir, f"{font_id}_{text_code}_{char_index}.png")

        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    if img.size != image_size:  # 크기 확인
                        img = img.resize(image_size, Image.LANCZOS)
                        img.save(image_path)
                    results.append((image_path, filtername))
                continue
            except:
                os.remove(image_path)

        if not os.path.exists(font_path):
            continue

        image = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(image)

        try:
            font_size = int(min(image_size) * 0.7)
            font = ImageFont.truetype(font_path, font_size)
            while True:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                if text_width > image_size[0] * 0.8 or text_height > image_size[1] * 0.8:
                    font_size -= 1
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    break
                if font_size < 10:
                    break

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) // 2 - bbox[0],
                        (image_size[1] - text_height) // 2 - bbox[1])
            draw.text(position, text, font=font, fill='black')
            image.save(image_path)

            # 크기 조정 및 저장
            with Image.open(image_path) as img:
                if img.size != image_size:
                    img = img.resize(image_size, Image.LANCZOS)
                    img.save(image_path)
                results.append((image_path, filtername))
        except Exception as e:
            print(f"이미지 생성 오류: {e}")

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
        print("데이터셋 무결성 및 크기 검증 중...")
        count_before = len(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True))
        target_size = self.image_size

        corrupted_or_resized = 0
        for img_path in tqdm(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True), desc="이미지 확인 중"):
            try:
                with Image.open(img_path) as img:
                    if img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)
                        img.save(img_path)
                        corrupted_or_resized += 1
            except:
                os.remove(img_path)
                corrupted_or_resized += 1

        count_after = len(glob.glob(os.path.join(self.output_dir, "**/*.png"), recursive=True))
        print(f"데이터셋 검증: {corrupted_or_resized}개의 이미지 수정/제거. 크기: {count_before} -> {count_after}")

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
    """Validation and test data transforms with proper normalization"""
    return al.Compose([
        al.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    def __init__(self, image_paths, labels, transforms=None, target_size=(224, 224)):
        valid_items = []
        for path, label in zip(image_paths, labels):
            if os.path.exists(path):
                try:
                    with Image.open(path) as img:
                        img_format = img.format
                    valid_items.append((path, label))
                except:
                    pass
        self.image_paths, self.labels = zip(*valid_items) if valid_items else ([], [])
        self.transforms = transforms
        self.target_size = target_size

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            label = self.labels[idx]

            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            else:
                # CRITICAL FIX: Convert numpy array to float before creating tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            return image, label
        except Exception as e:
            print(f"Image loading error {image_path}: {e}")
            # Return a zero tensor with correct shape and TYPE
            image = np.zeros((3, *self.target_size), dtype=np.float32)
            return torch.tensor(image, dtype=torch.float32), self.labels[idx]

    def __len__(self):
        return len(self.image_paths)



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

        # 진행 상황을 표시하는 tqdm 사용
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            try:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                batch_correct = (preds == labels).sum().item()
                batch_total = labels.size(0) * labels.size(1)

                correct += batch_correct
                total += batch_total

                # 현재 배치의 정확도와 손실을 진행 바에 업데이트
                batch_acc = 100 * batch_correct / batch_total
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_acc:.2f}%",
                    'avg_loss': f"{running_loss / (train_pbar.n + 1):.4f}",
                    'avg_acc': f"{100 * correct / total:.2f}%"
                })
            except Exception as e:
                print(f"\nError in batch: {e}")
                continue

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 검증 데이터에 대한 진행 바
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                batch_correct = (preds == labels).sum().item()
                batch_total = labels.size(0) * labels.size(1)

                val_correct += batch_correct
                val_total += batch_total

                # 현재 배치의 검증 정확도와 손실을 진행 바에 업데이트
                batch_acc = 100 * batch_correct / batch_total
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_acc:.2f}%",
                    'avg_loss': f"{val_loss / (val_pbar.n + 1):.4f}",
                    'avg_acc': f"{100 * val_correct / val_total:.2f}%"
                })

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # 에포크 요약 출력
        print(f'\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 최고 성능 모델 저장 및 조기 종료 로직 (기존 코드와 동일)
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
            print(f'New best model saved with Val Acc: {val_acc:.2f}%')
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')

            if early_stopping_counter >= patience:
                print(f'Early stopping triggered. Best epoch was {best_epoch + 1} with Val Acc: {best_val_acc:.2f}%')
                break

        # 주기적인 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')

    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}')
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
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 다중 레이블 출력
        )

    def forward(self, x):
        return self.resnet(x)  # 로짓 반환 (시그모이드 이전)

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
            image = Image.new('RGB', (244, 244), color='black')
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


    csv_path = 'cnn-cate_filter_merged.csv'
    print(f"CSV 파일에서 폰트 데이터 로드 중: {csv_path}")

    # 폰트 데이터와 라벨 매핑 로드
    font_data, label_map = prepare_dynamic_dataset(csv_path)

    # 폰트 데이터와 라벨 매핑 로드
    # 라벨 맵 저장
    save_label_map(label_map)
    # 데이터셋 분할 - 폰트 데이터를 분할
    font_ids = list(range(len(font_data)))
    train_ids, temp_ids = train_test_split(font_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    # 분할된 폰트 데이터 생성
    train_font_data = [font_data[i] for i in train_ids]
    val_font_data = [font_data[i] for i in val_ids]
    test_font_data = [font_data[i] for i in test_ids]
    # For testing purposes, use a much smaller dataset
    train_font_data = train_font_data
    val_font_data = val_font_data
    test_font_data = test_font_data


    # 동적 데이터셋 생성
    train_transforms = get_train_transforms(use_gray=False)
    val_transforms = get_val_transforms()

    train_dataset = DynamicFontDataset(train_font_data, label_map, transforms=train_transforms, samples_per_font=200)
    val_dataset = DynamicFontDataset(val_font_data, label_map, transforms=val_transforms, samples_per_font=200)
    test_dataset = DynamicFontDataset(test_font_data, label_map, transforms=val_transforms, samples_per_font=200)

    # 출력 메시지에서 동적으로 계산된 값 사용
    print(f"Train dataset size: {len(train_dataset)} images ({len(train_font_data)} fonts)")
    print(f"Validation dataset size: {len(val_dataset)} images ({len(val_font_data)} fonts)")
    print(f"Test dataset size: {len(test_dataset)} images ({len(test_font_data)} fonts)")

    # DataLoader 생성
    # 클래스 균형 맞추기는 여기서는 적용하지 않음 (동적 생성이기 때문에 복잡해짐)
    train_loader = DataLoader(
        train_dataset,
        batch_size=460,  # 더 작은 배치 사이즈 사용 (이미지 생성 시간 고려)
        shuffle=True,
        num_workers=28,
        pin_memory=True,
        multiprocessing_context='spawn'
    )

    val_loader = DataLoader(val_dataset, batch_size=460, shuffle=False, num_workers=28, pin_memory=True,
                            multiprocessing_context='spawn')
    test_loader = DataLoader(test_dataset, batch_size=460, shuffle=False, num_workers=28, pin_memory=True,
                             multiprocessing_context='spawn')

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
    criterion = nn.BCEWithLogitsLoss()  # 다중 레이블 손실 함수
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