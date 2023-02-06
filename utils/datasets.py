import os

import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from skimage.io import imread



class ImageDataset(Dataset):

    def __init__(self, folder, transform=None, size=None):
        self.images_folder = os.path.join(folder, 'inputs')
        self.semantic_annotations_folder = os.path.join(folder, 'semantic_annotations')
        self.depth_annotations_folder = os.path.join(folder, 'depth_annotations')
        self.images_paths = os.listdir(self.images_folder)
        self.semantic_annotations_paths = os.listdir(self.semantic_annotations_folder)
        self.depth_annotations_paths = os.listdir(self.depth_annotations_folder)
        if size is not None:
            self.images_paths = self.images_paths[:size]
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_name = self.images_paths[idx]
        image_path = os.path.join(self.images_folder, img_name)
        semantic_annotation_path = os.path.join(self.semantic_annotations_folder, img_name)
        depth_annotation_path = os.path.join(self.depth_annotations_folder, img_name)

        image = read_image(image_path).float()
        semantic_annotation = read_image(semantic_annotation_path).float()
        depth_annotation = imread(depth_annotation_path).astype(np.float32)
        depth_annotation = torch.from_numpy(depth_annotation)

        if self.transform:
            image = self.transform(image)

        return image, semantic_annotation, depth_annotation