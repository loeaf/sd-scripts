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


# Font Image Generator
@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (128, 128)  # Reduced size for faster training
    output_dir: str = "datasets/font_dataset"
    korean_unicode_file: str = "union_korean_unicodes.json"
    sample_texts: List[str] = None
    chars_per_image: int = 1  # Number of characters to include in each image

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

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

        if self.sample_texts is None:
            # Default Korean sample texts (used if not generating from Unicode)
            self.sample_texts = [
                "안녕하세요",
                "폰트 분류",
                "딥러닝",
                "인공지능",
                "한글 폰트"
            ]

    def create_font_image(self, text: str, font_path: str, font_id: int, family_name: str) -> str:
        """Generate an image with text rendered using the specified font"""
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)

        try:
            font_size = 1
            font = ImageFont.truetype(font_path, font_size)

            # Find the maximum font size that fits nicely
            while True:
                next_font = ImageFont.truetype(font_path, font_size + 1)
                bbox = draw.textbbox((0, 0), text, font=next_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width > self.image_size[0] * 0.8 or text_height > self.image_size[1] * 0.8:
                    break

                font = next_font
                font_size += 1

            # Center the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((self.image_size[0] - text_width) // 2 - bbox[0],
                        (self.image_size[1] - text_height) // 2 - bbox[1])

            # Draw the text on the image
            draw.text(position, text, font=font, fill='black')

            # Save the image
            os.makedirs(os.path.join(self.output_dir, str(font_id)), exist_ok=True)

            # Create a safe filename by replacing any problematic characters
            safe_text = ''.join([c if c.isalnum() else '_' for c in text])
            image_path = os.path.join(self.output_dir, str(font_id), f"{family_name}_{safe_text}.png")
            image.save(image_path)

            return image_path
        except Exception as e:
            print(f"Error processing font '{font_path}' with text '{text}': {str(e)}")
            return None

    def create_unicode_font_image(self, char: str, font_path: str, font_id: int, family_name: str,
                                  char_index: int) -> str:
        """Generate an image with a single Unicode character using the specified font"""
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)

        try:
            # Use a larger starting font size for single characters
            font_size = 10
            font = ImageFont.truetype(font_path, font_size)

            # Find the maximum font size that fits nicely
            while True:
                next_font = ImageFont.truetype(font_path, font_size + 5)  # Increment by 5 for faster sizing
                bbox = draw.textbbox((0, 0), char, font=next_font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]

                if char_width > self.image_size[0] * 0.7 or char_height > self.image_size[1] * 0.7:
                    break

                font = next_font
                font_size += 5

            # Fine-tune the final font size
            while True:
                next_font = ImageFont.truetype(font_path, font_size + 1)
                bbox = draw.textbbox((0, 0), char, font=next_font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]

                if char_width > self.image_size[0] * 0.8 or char_height > self.image_size[1] * 0.8:
                    break

                font = next_font
                font_size += 1

            # Center the character
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            position = ((self.image_size[0] - char_width) // 2 - bbox[0],
                        (self.image_size[1] - char_height) // 2 - bbox[1])

            # Draw the character on the image
            draw.text(position, char, font=font, fill='black')

            # Save the image
            os.makedirs(os.path.join(self.output_dir, str(font_id)), exist_ok=True)

            # Use Unicode code point for filename
            unicode_code = f"U{ord(char):04X}"
            image_path = os.path.join(self.output_dir, str(font_id), f"{family_name}_{unicode_code}_{char_index}.png")
            image.save(image_path)

            return image_path
        except Exception as e:
            print(f"Error processing font '{font_path}' with character '{char}' (U+{ord(char):04X}): {str(e)}")
            return None

    def generate_dataset_from_csv(self, csv_path: str, num_samples_per_font: int = 50, use_unicode: bool = True):
        """Generate a dataset from the font CSV file using either preset texts or Unicode characters"""
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create a label mapping and save it
        unique_families = df['FamilyName'].unique()
        label_map = {family: idx for idx, family in enumerate(unique_families)}

        with open('label.txt', 'w', encoding='utf-8') as f:
            for family, idx in label_map.items():
                f.write(f"{idx},{family}\n")

        print(f"Label mapping saved to label.txt with {len(label_map)} classes")

        # Generate images for each font
        image_paths = []
        labels = []

        for _, row in df.iterrows():
            font_id = row['font_id']
            font_path = row['FilePath']
            family_name = row['FamilyName']

            if use_unicode:
                # Use random Korean Unicode characters
                if len(self.korean_chars) <= num_samples_per_font:
                    chars_to_use = self.korean_chars  # Use all available if fewer than requested
                else:
                    chars_to_use = random.sample(self.korean_chars, num_samples_per_font)

                for i, char in enumerate(chars_to_use):
                    img_path = self.create_unicode_font_image(char, font_path, font_id, family_name, i)
                    if img_path:
                        image_paths.append(img_path)
                        labels.append(label_map[family_name])
            else:
                # Use preset sample texts
                texts_to_use = self.sample_texts
                if len(texts_to_use) > num_samples_per_font:
                    texts_to_use = random.sample(texts_to_use, num_samples_per_font)

                for text in texts_to_use:
                    img_path = self.create_font_image(text, font_path, font_id, family_name)
                    if img_path:
                        image_paths.append(img_path)
                        labels.append(label_map[family_name])

        print(f"Generated {len(image_paths)} images for {len(label_map)} font families")
        return image_paths, labels, label_map


# Custom Dataset class
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
            nn.Linear(256 * 8 * 8, 512),  # Assuming 128x128 input, after 4 max-pooling layers: 8x8
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
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
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

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

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
    return model


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
        chars_per_image=1
    )
    csv_path = 'cnn-cate_filter_merged.csv'

    print("Generating dataset from fonts using Korean Unicode characters...")
    image_paths, labels, label_map = generator.generate_dataset_from_csv(
        csv_path,
        num_samples_per_font=50,  # Number of Unicode characters to sample per font
        use_unicode=True  # Use Unicode characters instead of preset texts
    )

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

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
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        patience=5,  # Early stopping patience
        device=device
    )

    print("Training completed!")

    # Save the final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'label_map': label_map,
    }, 'font_classifier_final.pth')

    print("Model saved to font_classifier_final.pth")


if __name__ == "__main__":
    main()