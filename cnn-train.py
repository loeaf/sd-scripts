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
    """Generate an image with a single Unicode character using the specified font"""
    char, font_path, font_id, filternames, char_index, image_size, output_dir = params

    # 결과를 저장할 리스트
    results = []

    # 각 필터네임에 대해 이미지 생성
    for filtername in filternames:
        # 필터네임을 폴더 이름으로 안전하게 변환
        safe_filtername = filtername.replace('/', '_').replace(' ', '_')

        # Skip if the file already exists to avoid redundant work
        unicode_code = f"U{ord(char):04X}"
        out_dir = os.path.join(output_dir, safe_filtername)
        os.makedirs(out_dir, exist_ok=True)
        image_path = os.path.join(out_dir, f"{font_id}_{unicode_code}_{char_index}.png")

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

        try:
            # Use a fixed font size instead of adaptive sizing for speed
            font_size = int(min(image_size) * 0.7)
            font = ImageFont.truetype(font_path, font_size)

            # Center the character
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            position = ((image_size[0] - char_width) // 2 - bbox[0],
                        (image_size[1] - char_height) // 2 - bbox[1])

            # Draw the character on the image
            draw.text(position, char, font=font, fill='black')

            # Save the image
            image.save(image_path)

            # Verify the image was saved correctly
            try:
                with Image.open(image_path) as img:
                    img_format = img.format
                results.append((image_path, filtername))
            except Exception:
                # Remove corrupted file
                try:
                    os.remove(image_path)
                except:
                    pass

        except Exception as e:
            pass

    return results


# Font Image Generator
@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (128, 128)  # Reduced size for faster training
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

            # Use a fixed set of characters for all fonts to reduce variability
            if len(self.korean_chars) <= self.num_samples_per_font:
                chars_to_use = self.korean_chars
            else:
                # Use the same random seed for all fonts to ensure consistency
                random.seed(42)
                chars_to_use = random.sample(self.korean_chars, self.num_samples_per_font)

            for i, char in enumerate(chars_to_use):
                # Include image_size and output_dir in the parameters for the standalone function
                tasks.append((char, font_path, font_id, filternames, i, self.image_size, self.output_dir))

        # Generate images in parallel using multiprocessing
        print(f"Generating images for {len(tasks)} tasks using {self.num_processes} processes...")

        image_paths = []
        labels = []

        # Use multiprocessing pool with chunking for better performance
        with Pool(processes=self.num_processes) as pool:
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

            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                early_stopping_counter = 0
                torch.save(model.state_dict(), 'best_filter_classifier.pth')
                print(f'Model saved with Val Acc: {val_acc:.2f}%')
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


# Main execution function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the generator and create dataset
    generator = FontImageGenerator(
        korean_unicode_file="union_korean_unicodes.json",
        num_samples_per_font=20,
        num_processes=cpu_count(),  # Use all available CPU cores
        clean_output_dir=False  # Set to True to start fresh
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

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = FontDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FontDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = FontDataset(test_paths, test_labels, transform=val_transform)

    # Check if datasets are valid
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Empty dataset after filtering invalid images.")
        return

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    num_classes = len(label_map)
    model = FilterClassifierCNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model
    print("Starting training...")
    try:
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=50,
            patience=10,
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