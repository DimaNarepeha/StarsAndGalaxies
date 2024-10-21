import os

import PIL
import pandas as pd
import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_id_of_class_by_label


class CustomDataset(Dataset):

    def __init__(self, csv_file_labels, csv_all_classes, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            num_classes (int): Number of distinct classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.full_annotations = pd.read_csv(csv_file_labels)
        self.all_classes = pd.read_csv(csv_all_classes)
        self.root_dir = root_dir
        self.num_classes = self.all_classes['class_name'].size
        self.transform = transform

        # Filter annotations to only include images that exist in root_dir
        self.annotations = self.full_annotations[self.full_annotations['id'].apply(self.image_exists)]
        self.annotations.reset_index(drop=True, inplace=True)

    def image_exists(self, file_name):
        img_path = os.path.join(self.root_dir, file_name + '.jpg')
        return os.path.exists(img_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + '.jpg')
        image = PIL.Image.open(img_name).convert('RGB')  # Convert image to RGB
        label = self.annotations.iloc[idx, 1]  # Assuming the label is in the second column

        # # One-hot encode the label
        # one_hot_label = torch.zeros(self.num_classes, dtype=torch.float32)
        # # get id of class
        id_of_class = get_id_of_class_by_label(label, self.all_classes)
        # one_hot_label[id_of_class] = 1

        if self.transform:
            image = self.transform(image)

        return image, id_of_class
