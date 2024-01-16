import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Define the custom random dataset
class CustomRandomDataset(Dataset):
    def __init__(self, num_samples, num_classes, image_size):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
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

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define the landmarks for images
landmarks = {
    0: "Landmark for class 0",
    1: "Landmark for class 1",
    2: "Landmark for class 2",
    3: "Landmark for class 3",
    4: "Landmark for class 4",
    5: "Landmark for class 5"
}

# Parameters
num_samples = 1000
num_classes = 6
image_size = 64
batch_size = 32
num_epochs = 10

# Create the custom dataset
custom_dataset = CustomRandomDataset(num_samples, num_classes, image_size)

# Create data loaders
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Initialize the CNN model
model = CNN(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Use landmarks for images
for label, landmark in landmarks.items():
    print(f"Landmark for class {label}: {landmark}")

#---------------------------------------

class AlexNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LeNet5(nn.Module):
    def __init__(self, num_classes=6):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120 * 53 * 53, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetModel(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)


#-------------------


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=30):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {val_loss/len(self.val_loader)}, Validation Accuracy: {correct/total}')

    def grad_cam(self, input_tensor, target_class=None):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Get the output from the final convolutional layer
        final_conv_layer = None
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                final_conv_layer = layer

        # Get the gradients with respect to the output of the final convolutional layer
        self.model.zero_grad()
        output = self.model(input_tensor.to(device))
        if target_class is None:
            target_class = torch.argmax(output)
        output[:, target_class].backward()

        # Get the feature map from the final convolutional layer
        feature_maps = final_conv_layer.weight.grad
        alpha = torch.mean(feature_maps, dim=(2, 3), keepdim=True)

        # Perform weighted combination to get the heatmap
        heatmap = torch.sum(alpha * feature_maps, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        return heatmap

    def overlay_heatmap(self, image, heatmap):
        heatmap = heatmap.squeeze().cpu().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        return overlayed_image

    def evaluate_and_save_results(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        confusion_matrix
