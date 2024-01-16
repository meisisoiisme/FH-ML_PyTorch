import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import cv2  

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
        return transforms.ToTensor()(image.astype(np.uint8)), label

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes, image_size):
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

# Parameters
num_samples = 100
num_classes = 6
image_size = 64
batch_size = 32
num_epochs = 30

# Create the custom dataset
custom_dataset = CustomRandomDataset(num_samples, num_classes, image_size)

# Create data loaders
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Initialize the CNN model
model = CNN(num_classes, image_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Use landmarks for images
landmarks = {
    0: "Landmark for class 0",
    1: "Landmark for class 1",
    2: "Landmark for class 2",
    3: "Landmark for class 3",
    4: "Landmark for class 4",
    5: "Landmark for class 5"
}

for label, landmark in landmarks.items():
    print(f"Landmark for class {label}: {landmark}")
