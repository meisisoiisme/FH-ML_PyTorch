# Install required libraries
!pip install torch torchvision matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import random
import cv2
import csv

# Constants
NUM_CLASSES = 5
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_SAMPLES = 500
NUM_EPOCHS = 10
DATA_PATH = "./training_images"
LANDMARKS_CSV_PATH = "./training_images_with_landmarks/landmarks.csv"

# Function to create data loaders
def create_data_loaders(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Function to initialize different CNN models
def initialize_model(model_name, num_classes, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Modify the classifier for the new task
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels, landmarks in train_loader:
            inputs, labels, landmarks = inputs.to(device), labels.to(device), landmarks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, landmarks in val_loader:
                inputs, labels, landmarks = inputs.to(device), labels.to(device), landmarks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}')

        # Check if the current model has the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

    # Load the best model weights
    model.load_state_dict(best_model_state)

    return model

# Function to evaluate and save results
def evaluate_and_save_results(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, landmarks in test_loader:
            inputs, labels, landmarks = inputs.to(device), labels.to(device), landmarks.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# CustomRandomDataset class
class CustomRandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, num_classes, image_size, data_path, landmarks_csv_path):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data_path = data_path
        self.landmarks_csv_path = landmarks_csv_path
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # Generate random image data
            image = np.random.rand(image_size, image_size, 3) * 255  # Random RGB image
            label = random.randint(0, num_classes-1)  # Random label
            self.data.append(image)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return transforms.ToTensor()(image), label

# ImageGenerator class
class ImageGenerator:
    def __init__(self, output_dir, num_classes=5, images_per_class=100, image_size=(224, 224), noise_factor=20):
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self.image_size = image_size
        self.noise_factor = noise_factor

    def generate_images(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for class_idx in range(self.num_classes):
            class_dir = os.path.join(self.output_dir, f"class_{class_idx}")
            os.makedirs(class_dir, exist_ok=True)

            for i in range(self.images_per_class):
                # Create a base image for the class
                base_image = np.random.randint(0, 256, size=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

                # Introduce minor differences (noise) to simulate variations in the same class
                noisy_image = base_image + np.random.randint(-self.noise_factor, self.noise_factor + 1, size=base_image.shape)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

                # Convert NumPy array to PIL Image
                pil_image = Image.fromarray(noisy_image)

                # Save the image to the corresponding class directory
                image_path = os.path.join(class_dir, f"image_{i}.png")
                pil_image.save(image_path)

        print(f"{self.num_classes} classes with {self.images_per_class} images each generated and saved in {self.output_dir}")

# LandmarksImageGenerator class
class LandmarksImageGenerator(ImageGenerator):
    def __init__(self, output_dir, num_classes=5, images_per_class=100, image_size=(224, 224), noise_factor=20, num_landmarks=5):
        super().__init__(output_dir, num_classes, images_per_class, image_size, noise_factor)
        self.num_landmarks = num_landmarks

    def generate_images_with_landmarks(self):
        super().generate_images()

        csv_file_path = os.path.join(self.output_dir, "landmarks.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image Path', 'Class', 'Landmark 1 X', 'Landmark 1 Y', 'Landmark 2 X', 'Landmark 2 Y', '...'])

            for class_idx in range(self.num_classes):
                class_dir = os.path.join(self.output_dir, f"class_{class_idx}")

                for i in range(self.images_per_class):
                    # Add landmarks to the image
                    landmarks = []
                    for landmark_idx in range(self.num_landmarks):
                        landmark_x = random.randint(0, self.image_size[0])
                        landmark_y = random.randint(0, self.image_size[1])
                        landmarks.extend([landmark_x, landmark_y])

                    # Write landmark information to CSV
                    image_path = os.path.join(class_dir, f"image_{i}.png")
                    csv_writer.writerow([image_path, class_idx] + landmarks)

        print(f"{self.num_classes} classes with {self.images_per_class} images each generated and saved in {self.output_dir}")
        print(f"Landmark information saved in {csv_file_path}")

# Generate and save images
output_directory = "./training_images"
image_gen = ImageGenerator(output_directory, num_classes=NUM_CLASSES, images_per_class=100, noise_factor=20)
image_gen.generate_images()

output_directory_landmarks = "./training_images_with_landmarks"
landmarks_gen = LandmarksImageGenerator(output_directory_landmarks, num_classes=NUM_CLASSES, images_per_class=100, noise_factor=20, num_landmarks=5)
landmarks_gen.generate_images_with_landmarks()

# Create the custom datasets
custom_dataset = CustomRandomDataset(NUM_SAMPLES, NUM_CLASSES, IMAGE_SIZE, DATA_PATH, LANDMARKS_CSV_PATH)
custom_val_dataset = CustomRandomDataset(NUM_SAMPLES // 5, NUM_CLASSES, IMAGE_SIZE, DATA_PATH, LANDMARKS_CSV_PATH)  # Adjust as needed
custom_test_dataset = CustomRandomDataset(NUM_SAMPLES // 5, NUM_CLASSES, IMAGE_SIZE, DATA_PATH, LANDMARKS_CSV_PATH)  # Adjust as needed

# Create data loaders
train_loader = create_data_loaders(custom_dataset)
val_loader = create_data_loaders(custom_val_dataset, shuffle=False)
test_loader = create_data_loaders(custom_test_dataset, shuffle=False)

# Initialize the model (Choose from 'resnet18', 'resnet34', 'densenet121')
model_name = 'resnet18'
model = initialize_model(model_name, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

# Evaluate and save results
evaluate_and_save_results(trained_model, test_loader)
