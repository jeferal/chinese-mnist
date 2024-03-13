"""
    This module implements the dataset and data loaders abstractions
"""

import os
import random

import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision.io import read_image

import pandas as pd

import cv2
import numpy as np


class ChineseMnistDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self._img_labels = pd.read_csv(annotations_file)
        self._img_dir = img_dir
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._img_labels)

    def __getitem__(self, idx):
        # Fields needed to get the path
        suite_id = self._img_labels['suite_id'][idx]
        sample_id = self._img_labels["sample_id"][idx]
        code = self._img_labels["code"][idx]

        label = code - 1  # Convert to 0 indexed 

        img_name = f"input_{suite_id}_{sample_id}_{code}.jpg"

        img_path = os.path.join(self._img_dir, img_name)
        image = read_image(img_path)

        if self._transform:
            image = self._transform(image)
        if self._transform:
            label = self._transform(label)
        return image, label


if __name__ == "__main__":
    # Test the data loader
    annotations_file = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/chinese_mnist.csv"
    img_dir = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/data"

    dataset = ChineseMnistDataset(annotations_file, img_dir)

    # Split the dataset
    split = random_split(dataset, [10000, 2500, 2500])

    # It returns a list with 3 elements
    training_data = split[0]
    validation_data = split[1]
    test_data = split[2]

    print(f"This is the length of the dataset: {len(dataset)}")
    
    # Show a random image
    random_index = random.randint(0, len(dataset))

    print(f"Asking for image {random_index}")
    image, label = dataset[random_index]
    window_name = f"This has the lable {label}"

    image_array = image.permute(1, 2, 0).cpu().numpy()
    image_array = (image_array * 255).astype(np.uint8)

    cv2.imshow(window_name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
