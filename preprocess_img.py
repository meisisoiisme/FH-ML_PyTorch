import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from google.colab import files

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch, image

def grad_cam(model, input_tensor, target_class=None):
    model.eval()
    device = next(model.parameters()).device

    # Get the output from the final convolutional layer
    final_conv_layer = None
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            final_conv_layer = layer

    # Get the gradients with respect to the output of the final convolutional layer
    model.zero_grad()
    output = model(input_tensor.to(device))
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

def overlay_heatmap(image, heatmap):
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlayed_image

def main():
    # Upload the training data
    data_dir = files.upload()

    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
    ])

    # Create dataset with target_transform to ensure consistent classes
    dataset = datasets.ImageFolder(root='/content/data', transform=data_transform, target_transform=lambda x: x)

    # Split dataset into training and validation sets
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Access the classes attribute from the dataset of the Subset
    num_classes = len(train_set.dataset.classes)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

    # Print some information
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")

    # Choose the model (replace with your model)
    model = models.alexnet(pretrained=True)
    model.eval()

    # Grad-CAM for the predicted class
    for i, (inputs, labels) in enumerate(val_loader):
        input_tensor = inputs[0].unsqueeze(0)  # Take the first sample for simplicity
        original_image = transforms.ToPILImage()(inputs[0])

        # Grad-CAM for the predicted class
        heatmap = grad_cam(model, input_tensor)

        # Overlay heatmap on the original image
        overlayed_image = overlay_heatmap(original_image, heatmap)

        # Display the original image, heatmap, and overlayed image
        plt.subplot(131)
        plt.imshow(original_image)
        plt.title('Original Image')

        plt.subplot(132)
        plt.imshow(heatmap.squeeze(), cmap='hot')
        plt.title('Heatmap')

        plt.subplot(133)
        plt.imshow(overlayed_image)
        plt.title('Overlayed Image')

        plt.show()

    # Continue with training and evaluation if needed

if __name__ == "__main__":
    main()
