import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageFile
import torch.nn as nn
import torch.optim as optim
import glob
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True 
# Dataset class for Udacity simulator
class DrivingDataset(Dataset):
    def __init__(self, base_dir, label_file, transform=None):
        self.transform = transform
        self.samples = []

        # Load labels from txt file into dict
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                filename, value = line.strip().split()
                self.labels[filename] = float(value)   # convert to float (or int if categorical)

        # Collect all image paths
        files = os.listdir(base_dir)
        for fl in files:
            if fl.endswith(".jpg") and fl in self.labels:
                self.samples.append(os.path.join(base_dir, fl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        filename = os.path.basename(img_path)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = Image.fromarray(image)
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label from dict
        label = self.labels[filename]

        return image, label



class DrivingOODDataset(Dataset):
    def __init__(self, base_dir, label, transform=None):
        print(base_dir)
        self.transform = transform
        self.samples = []
        self.label = label

        # Load labels from txt file into dict
        # Collect all image paths
        files = os.listdir(base_dir)
        for fl in files:
            if fl.endswith(".jpg") or fl.endswith(".png"):
                self.samples.append(os.path.join(base_dir, fl))
        if label == 1:
            print("Attack Dataset Length:", len(self.samples))
        elif label == 0:
            print("Normal Dataset Length:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        image = Image.fromarray(image)

        image = self.transform(image)   

        return image, self.label


class DrivingOODDatasetNpy(Dataset):
    def __init__(self, base_dir, label, transform=None):
        print(base_dir)
        self.transform = transform
        self.samples = []
        self.label = label

        # Load labels from txt file into dict
        # Collect all image paths
        files = os.listdir(base_dir)
        for fl in files:
            if fl.endswith(".npy"):
                self.samples.append(os.path.join(base_dir, fl))
        if label == 1:
            print("Attack Dataset Length NPY:", len(self.samples))
        elif label == 0:
            print("Normal Dataset Length NPY:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image)

        return image, self.label
