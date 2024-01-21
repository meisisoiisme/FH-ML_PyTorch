import os
import random
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, num_samples, num_classes, image_size, image_path=None, labels_path=None, landmarks_csv_path=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data = []
        self.labels = []
        self.image_path = image_path
        self.labels_path = labels_path
        self.landmarks_csv_path = landmarks_csv_path
        self.num_landmarks = 5  # Assuming the default number of landmarks
        self.landmarks = self.load_landmarks()

        # Check if image_path, labels_path, and landmarks_csv_path are provided
        if image_path is None or labels_path is None or landmarks_csv_path is None:
            raise ValueError("All three paths (image_path, labels_path, landmarks_csv_path) must be provided.")

        # Check if the provided paths exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels path not found: {labels_path}")

        if not os.path.exists(landmarks_csv_path):
            raise FileNotFoundError(f"Landmarks CSV path not found: {landmarks_csv_path}")

        # Load data
        self.load_data()

    def load_data(self):
        if self.image_path is not None and self.labels_path is not None:
            for label in range(self.num_classes):
                label_path = os.path.join(self.image_path, f"class_{label}")
                if os.path.exists(label_path):
                    class_images = [os.path.join(label_path, image_file) for image_file in os.listdir(label_path)]
                    if len(class_images) == 0:
                        print(f"Warning: Class {label} has no images.")
                        continue

                    self.data.extend(class_images)
                    self.labels.extend([label] * len(class_images))

        # Debugging print statement
        # print(f"Available image paths: {self.data}")

        # Shuffle the data after loading
        random.shuffle(self.data)
        random.shuffle(self.labels)

    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            return transforms.ToTensor()(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_landmarks(self):
        landmarks = np.zeros((len(self.data), 2 * self.num_landmarks), dtype=np.float32)
        # Debugging print statement
        # print(f"Available image paths: {self.data}")
        try:
            with open(self.landmarks_csv_path, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                header = next(csv_reader)  # Skip header row

                # Debugging print statement
                # print(f"CSV Header: {header}")

                for i, row in enumerate(csv_reader):
                    if len(row) < 2 * self.num_landmarks + 2:
                        print(f"Error: Incomplete row in CSV at index {i + 1}. Expected {2 * self.num_landmarks + 2} columns, found {len(row)}.")
                        continue

                    image_path, _, *landmark_values = row

                    # Construct the full image path by joining data_path and image_path
                    # full_image_path = os.path.join(self.image_path, image_path)

                    # Debugging print statements
                    # print(f"Checking image path: {image_path}")
                    # print(f"Available image paths: {self.data}")

                    # Check if the full_image_path is in the list of available image paths
                    if image_path not in self.data:
                        # print(f"Warning: Image path {image_path} not in list of available image paths.")
                        continue

                    image_index = self.data.index(image_path)
                    # print(f"image path: {image_path}")
                    # print(f"image_index: {image_index}")
                    # Debugging print statement
                    # print(f"Landmark values: {landmark_values}")

                    landmarks[image_index] = list(map(float, landmark_values))

        except Exception as e:
            print(f"Error while reading landmarks CSV: {e}")

        return torch.tensor(landmarks)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            image_path = self.data[index]
            label = self.labels[index]
            image = self.load_image(image_path)
            landmarks = self.landmarks[index]
            return image, label, landmarks
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            return torch.zeros((3, self.image_size, self.image_size)), -1, torch.zeros((2 * self.num_landmarks))
