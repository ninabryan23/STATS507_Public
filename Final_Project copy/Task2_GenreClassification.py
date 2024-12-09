#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
from urllib import request
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import ssl
import certifi

# Setup SSL Context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Constants
NUM_DOWNLOADS_TRAIN = 1600
NUM_DOWNLOADS_TEST = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16

# Argument Parser
parser = ArgumentParser()
parser.add_argument(
    'output_dirpath',
    type=str,
    help='Output directory for images'
)
args = parser.parse_args()

# Helper: Download Images
def download_images(csv_filepath, output_dir, num_downloads=None):
    header_names = ['Amazon ID (ASIN)', 'Filename', 'Image URL', 'Title', 'Author', 'Category ID', 'Category']
    csv = pd.read_csv(csv_filepath, delimiter=",", header=None, names=header_names, encoding='latin1')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print(f'[Downloading images into "{output_dir}"]')
    for i in tqdm(range(min(num_downloads or len(csv), len(csv))), desc="Downloading Images"):
        row = csv.iloc[i]
        filename = row['Filename']
        url = row['Image URL']
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            try:
                response = request.urlopen(url, context=ssl_context)
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            except Exception as e:
                print(f"Failed to download {url}: {e}")

# Helper: Validate and Load Image
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Invalid image file {image_path}: {e}")
        return None

# Helper: Create Dataset
def create_dataset(image_dir, labels_map):
    dataset = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        if img_file in labels_map:
            image = load_image(img_path)
            if image:
                dataset.append((img_path, labels_map[img_file]))
    return dataset


# Initialize Training Data
train_csv_path = "data/book30-listing-train.csv"
train_labels_path = "data/bookcover30-labels-train.txt"
train_output_dir = os.path.join(args.output_dirpath, "train")
download_images(train_csv_path, train_output_dir, num_downloads=NUM_DOWNLOADS_TRAIN)

train_labels_df = pd.read_csv(train_labels_path, delim_whitespace=True, header=None, names=['Filename', 'Category ID'])
train_labels_map = train_labels_df.set_index('Filename')['Category ID'].to_dict()
train_dataset = create_dataset(train_output_dir, train_labels_map)

# Compute Class Weights
class_weights = torch.tensor(
    [1.0 / train_labels_df['Category ID'].value_counts().get(i, 1) for i in range(len(train_labels_df['Category ID'].unique()))],
    dtype=torch.float
).to(DEVICE)

# Initialize Testing Data
test_csv_path = "data/book30-listing-test.csv"
test_labels_path = "data/bookcover30-labels-test.txt"
test_output_dir = os.path.join(args.output_dirpath, "test")
download_images(test_csv_path, test_output_dir, num_downloads=NUM_DOWNLOADS_TEST)

test_labels_df = pd.read_csv(test_labels_path, delim_whitespace=True, header=None, names=['Filename', 'Category ID'])
test_labels_map = test_labels_df.set_index('Filename')['Category ID'].to_dict()
test_dataset = create_dataset(test_output_dir, test_labels_map)

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Augmentation: Rotate up to ±15 degrees
    transforms.RandomCrop(224, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loss Function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training Function with Scheduler and Freezing
def train_model_with_freezing_and_scheduler(model, dataset, num_epochs=20, unfreeze_after=5):
    """Train the model with freezing, unfreezing, and learning rate scheduler."""
    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        if epoch == unfreeze_after:
            # Unfreeze all layers after `unfreeze_after` epochs
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model.train()
        total_loss = 0
        for img_path, label in dataset:
            image = load_image(img_path)
            if image is None:
                continue
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            label_tensor = torch.tensor([label], device=DEVICE)

            optimizer.zero_grad()
            output = model(image_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# Helper: Evaluate Model
def evaluate_model(model, dataset):
    model.eval()
    true_labels, predicted_labels = [], []
    for img_path, label in dataset:
        image = load_image(img_path)
        if image is None:
            continue
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        true_labels.append(label)
        predicted_labels.append(predicted.item())
    print("\nAccuracy Metrics:")
    print(f"Overall Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))


# Model Initialization
try:
    # Load Pre-trained ResNet50 Model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
except Exception as e:
    print(f"Failed to load weights: {e}")
    weights_path = "resnet50-11ad3fa6.pth"
    model = resnet50()
    model.load_state_dict(torch.load(weights_path))

# Adjust Final Fully Connected Layer
model.fc = torch.nn.Linear(model.fc.in_features, len(train_labels_df['Category ID'].unique()))
model.to(DEVICE)

# Train Model
train_model_with_freezing_and_scheduler(model, train_dataset, num_epochs=20, unfreeze_after=5)

# Evaluate Model
evaluate_model(model, test_dataset)