from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from torchvision import transforms
from PIL import Image

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=train_mean, std=train_std),
                                              ])
        self._train_transform = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.RandomHorizontalFlip(p=0.4),
                                                    transforms.RandomVerticalFlip(p=0.4),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(train_mean, train_std)
                                                    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "val":
            data = self.data.iloc[idx]
            image = Image.open(str(data['filename'])).convert('L')  # Open image and convert to grayscale
            image = image.convert('RGB')  # Convert to RGB
            image = np.array(image)
            label = np.array([data['crack'], data['inactive']])
            image = self._transform(image)
            return image, label
        if self.mode == "train":
            data = self.data.iloc[idx]
            image = Image.open(str(data['filename'])).convert('L')  # Open image and convert to grayscale
            image = image.convert('RGB')  # Convert to RGB
            image = np.array(image)
            label = np.array([data['crack'], data['inactive']])
            image = self._train_transform(image)
            return image, label
