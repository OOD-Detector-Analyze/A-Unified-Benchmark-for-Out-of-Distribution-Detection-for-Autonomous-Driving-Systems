import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from DataPreprocess.attacks import fgsm_attack, pgd_attack
import re
# Dataset class for Udacity simulator

class UdacityImageDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.samples = []

        data = ["track1", "track2", "track3"]
        mode = ["normal", "reverse", "sport_normal", "sport_reverse"]

        for track in data:
            for md in mode:
                folder = os.path.join(base_dir, track, md)

                csv_path = os.path.join(folder, "driving_log.csv")
                if not os.path.exists(csv_path):
                    continue  # skip if mode folder or CSV doesn't exist
                try:
                    df = pd.read_csv(csv_path, header=None)
                    img_paths = df.iloc[:, 0].values  # center images

                    for p in img_paths:
                        ab_path =os.path.join(base_dir, p)
                        # Extract relative path after 'track'
                        if os.path.exists(ab_path):
                            self.samples.append(ab_path)

                except Exception as e:
                    print(f"❌ Failed reading {csv_path}: {e}")

        print(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply augmentation
        # image = random_flip(image)
        # image = random_translate(image, range_x=100, range_y=10)
        # image = random_shadow(image)
        # image = random_brightness(image)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, 0


    
class UdacityImageTestDataset(Dataset):
    def __init__(self, base_dir, data, transform=None, mode='clean', fmodel=None, attack_type='FGSM', attack_frequency=2):
        self.transform = transform
        self.samples = []
        self.mode = mode
        self.fmodel = fmodel
        self.attack_type = attack_type
        self.attack_frequency = attack_frequency

        for track in data:
            folder = os.path.join(base_dir, track)
            image_path = os.path.join(folder, "IMG")
            if not os.path.exists(image_path):
                continue  # skip if folder doesn't exist
            try:
                images_list = os.listdir(image_path)
                for p in images_list:
                    ab_path = os.path.join(image_path, p)
                    if os.path.exists(ab_path):
                        self.samples.append(ab_path)
            except Exception as e:
                print(f"❌ Failed reading {image_path}: {e}")



        print(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def apply_attack(self, image_tensor):
        adv_img, _ = fgsm_attack(self.fmodel, image_tensor)
        return adv_img
      

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = 0

        return image, label
    

class UdacityImageAttackDataset(Dataset):
    def __init__(self, base_dir, data, transform=None):
        self.transform = transform
        self.samples = []
        print(base_dir, data)
        for track in data:
            folder = os.path.join(base_dir, track)
            image_path = os.path.join(folder, "IMG")
            if not os.path.exists(image_path):
                continue  # skip if folder doesn't exist
            try:
                images_list = os.listdir(image_path)
                for p in images_list:
                    ab_path = os.path.join(image_path, p)
                    if os.path.exists(ab_path):
                        self.samples.append(ab_path)
            except Exception as e:
                print(f"❌ Failed reading {image_path}: {e}")
        print(f"Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = 1

        return image, label

class AnomalImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.npy'))
        ]
        self.transform = transform
        print(len(self.image_paths), "anomal images loaded.")
        print(root_dir)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image)

        return image, 1