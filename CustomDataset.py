import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from skimage import io
from PIL import Image
from PIL import ImageFile

class ClassificationDataset(Dataset):
    def __init__(self ,image_paths, targets, resize=None, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = image.convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]),
                                 resample=Image.BILINEAR)

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        # torch expects C X H X W instead of H X W X C
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        image = torch.tensor(image, dtype=torch.float)
        targets = torch.tensor(targets, dtype=torch.long)

        return (image, targets)

