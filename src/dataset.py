import os
import random
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

def load_image_for_albumentations(path: str) -> np.ndarray:
    # This function is fine as it is.
    img = Image.open(path)
    if img.mode in ('RGB', 'L'):
        return np.array(img.convert('RGB'))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img)
    return np.array(background)


class ImageOrientationDataset(Dataset):
    def __init__(self, upright_dir):
        self.upright_dir = upright_dir
        self.image_files = []
        for root, _, files in os.walk(upright_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, filename))

        if not self.image_files:
            raise ValueError(f"No images found in the directory: {upright_dir}")

        # 1. Define the rotations and create a list of transform pipelines, one for each rotation.
        self.rotations = [0, -90, -180, -270] # Corresponds to labels 0, 1, 2, 3
        self.num_rotations = len(self.rotations)
        
        self.transforms = []
        for angle in self.rotations:
            # Create a full pipeline for this specific angle
            pipeline = A.Compose([
                A.Rotate(limit=(angle, angle), p=1.0, interpolation=Image.BILINEAR, border_mode=0),
                A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            self.transforms.append(pipeline)

    def __len__(self):
        return len(self.image_files) * self.num_rotations

    def __getitem__(self, idx):
        image_idx = idx // self.num_rotations
        rotation_idx = idx % self.num_rotations # This will be 0, 1, 2, or 3

        image_path = self.image_files[image_idx]
        label = rotation_idx

        try:
            image = load_image_for_albumentations(image_path)
        except Exception as e:
            logging.warning(f"Warning: Could not open {image_path}. Skipping. Error: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 2. Select the PRE-COMPILED transform from the list. This is extremely fast.
        transform = self.transforms[rotation_idx]

        transformed = transform(image=image)
        image_tensor = transformed['image']

        return image_tensor, torch.tensor(label, dtype=torch.long)