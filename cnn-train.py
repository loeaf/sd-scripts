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


# Function for multiprocessing - needs to be at module level
def create_unicode_font_image(params):
    """Generate an image with a single Unicode character using the specified font"""
    char, font_path, font_id, family_name, char_index, image_size, output_dir = params

    # Skip if the file already exists to avoid redundant work
    unicode_code = f"U{ord(char):04X}"
    out_dir = os.path.join(output_dir, str(font_id))
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, f"{family_name}_{unicode_code}_{char_index}.png")

    # Skip if file already exists (for resuming interrupted processing)
    if os.path.exists(image_path):
        return image_path, font_id, family_name

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

        return image_path, font_id, family_name
    except Exception as e:
        print(f"Error processing font '{font_path}' with character '{char}' (U+{ord(char):04X}): {str(e)}")
        return None, font_id, family_name


# Font Image Generator
@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (128, 128)  # Reduced size for faster training
    output_dir: str = "datasets/font_dataset"
    korean_unicode_file: str = "union_korean_unicodes.json"
    num_samples_per_font: int = 20  # Reduced from 50 to 20 for faster processing
    num_processes: int = None  # Will default to number of CPU cores

    def __post_init__(self):
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

    def generate_dataset_from_csv(self, csv_path: str):
        """Generate a dataset from the font CSV file using multiprocessing"""
        start_time = time.time()

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create a label mapping and save it
        unique_families = df['FamilyName'].unique()
        label_map = {family: idx for idx, family in enumerate(unique_families)}

        with open('label.txt', 'w', encoding='utf-8') as f:
            for family, idx in label_map.items():
                f.write(f"{idx},{family}\n")

        print(f"Label mapping saved to label.txt with {len(label_map)} classes")

        # Prepare tasks for parallel processing
        tasks = []

        for _, row in df.iterrows():
            font_id = row['font_id']
            font_path = row['FilePath']
            family_name = row['FamilyName']

            # Use a fixed set of characters for all fonts to reduce variability
            if len(self.korean_chars) <= self.num_samples_per_font:
                chars_to_use = self.korean_chars
            else:
                # Use the same random seed for all fonts to ensure consistency
                random.seed(42)
                chars_to_use = random.sample(self.korean_chars, self.num_samples_per_font)

            for i, char in enumerate(chars_to_use):
                # Include image_size and output_dir in the parameters for the standalone function
                tasks.append((char, font_path, font_id, family_name, i, self.image_size, self.output_dir))

        # Generate images in parallel using multiprocessing
        print(f"Generating {len(tasks)} images using {self.num_processes} processes...")

        image_paths = []
        labels = []

        # Use multiprocessing pool
        with Pool(processes=self.num_processes) as pool:
            # Use imap_unordered for better performance and wrap with tqdm for progress tracking
            results = list(
                tqdm(pool.imap_unordered(create_unicode_font_image, tasks), total=len(tasks), desc="Generating images"))

            # Process results
            for img_path, font_id, family_name in results:
                if img_path:
                    image_paths.append(img_path)
                    labels.append(label_map[family_name])

        end_time = time.time()
        print(
            f"Generated {len(image_paths)} images for {len(label_map)} font families in {end_time - start_time:.2f} seconds")
        return image_paths, labels, label_map


# On-the-fly Dataset class for better memory usage
class FontDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# CNN Model Architecture
class FontClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FontClassifierCNN, self).__init__()

        # Convolutional layers with reduced parameters
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fourth convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),  # Reduced from 512 to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5, device='cuda'):
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
            torch.save(model.state_dict(), 'best_font_classifier.pth')
            print(f'Model saved with Val Acc: {val_acc:.2f}%')
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')

        # Early stopping
        if early_stopping_counter >= patience:
            print(
                f'Early stopping triggered after epoch {epoch + 1}. Best epoch was {best_epoch + 1} with validation accuracy: {best_val_acc:.2f}%')
            break

    print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}')
    return model, history


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
        num_samples_per_font=20,  # Adjust as needed
        num_processes=cpu_count()  # Use all available CPU cores
    )
    csv_path = 'cnn-cate_filter_merged.csv'

    print(f"Generating dataset from fonts using {generator.num_processes} processes...")
    image_paths, labels, label_map = generator.generate_dataset_from_csv(csv_path)

    # Split the dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(1),
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize the model
    num_classes = len(label_map)
    model = FontClassifierCNN(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=15,
        patience=5,
        device=device
    )

    print("Training completed!")

    # Save the final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'label_map': label_map,
        'history': history
    }, 'font_classifier_final.pth')

    print("Model saved to font_classifier_final.pth")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    main()