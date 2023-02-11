import os
from typing import Any

import numpy as np
import torch
import wandb
from skimage.io import imread
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image

import configs.globals
import configs.wandb as wandb_config


class ImageDataset(Dataset):
    """
    Dataset for semantic segmentation and depth estimation for the SUADD Challenge.
    The transforms are applied at each iteration, so they should be random.
    """

    def __init__(self, folder, transform=None, store_images=False, size=None):
        """
        Create a dataset for the SUADD Challenge.
        :param folder: path to the folder containing the images, semantic annotations and depth annotations
        :param transform: transform to apply to the images
        :param store_images: if True, the images are stored in memory else they are read from the disk at each iteration
        :param size: number of images to use if None, all the images are used
        """
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.size = size
        self.store_images = store_images

        self.images_folder = os.path.join(folder, 'inputs')
        self.semantic_annotations_folder = os.path.join(folder, 'semantic_annotations')
        self.depth_annotations_folder = os.path.join(folder, 'depth_annotations')
        self.images_paths = os.listdir(self.images_folder)
        self.semantic_annotations_paths = os.listdir(self.semantic_annotations_folder)
        self.depth_annotations_paths = os.listdir(self.depth_annotations_folder)
        if size is not None:
            self.images_paths = self.images_paths[:size]
        self.transform = transform

        self.mean = None
        self.std = None
        if self.transform is not None:
            self.compute_mean_std()
            self.transform.set_mean_std(self.mean, self.std)

        if store_images:
            self.images = []
            self.semantic_annotations = []
            self.depth_annotations = []
            for i in range(len(self.images_paths)):
                img_name = self.images_paths[i]
                depth_annotation, image, semantic_annotation = self.load_datapoint(img_name)
                self.images.append(image)
                self.semantic_annotations.append(semantic_annotation)
                self.depth_annotations.append(depth_annotation)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):

        if self.train_idx is None:
            train = False
            validation = False
            test = False
        else:
            train = idx in self.train_idx
            validation = idx in self.val_idx
            test = idx in self.test_idx

        img_name = self.images_paths[idx]

        if self.store_images:
            depth_annotation = self.depth_annotations[idx]
            image = self.images[idx]
            semantic_annotation = self.semantic_annotations[idx]
        else:
            depth_annotation, image, semantic_annotation = self.load_datapoint(img_name)

        image_shape = image.shape

        if train or validation or test:
            image, semantic_annotation, depth_annotation = self.transform(image, semantic_annotation, depth_annotation,
                                                                          train=train)

        output = {
            "image": image,
            "semantic": semantic_annotation,
            "depth": depth_annotation,
            "image_shape": [image_shape],
            "image_name": img_name.replace(".png", ""),
        }

        return output

    def load_datapoint(self, img_name):
        """
        Load a single datapoint
        :param img_name: name of the image
        :return: the image, the semantic annotation and the depth annotation
        """
        image_path = os.path.join(self.images_folder, img_name)
        semantic_annotation_path = os.path.join(self.semantic_annotations_folder, img_name)
        depth_annotation_path = os.path.join(self.depth_annotations_folder, img_name)
        image = read_image(image_path).float()
        semantic_annotation = read_image(semantic_annotation_path).float()
        semantic_annotation[semantic_annotation == 255] = len(configs.globals.CLASSES) - 1
        depth_annotation = imread(depth_annotation_path).astype(np.float32)
        mask = depth_annotation == 0
        depth_annotation = (depth_annotation - 1.) / 128.
        depth_annotation[mask] = 0
        depth_annotation = torch.from_numpy(depth_annotation).unsqueeze(0)
        return depth_annotation, image, semantic_annotation

    def compute_mean_std(self):
        """
        Compute the mean and std of the training dataset
        :return:
        """

        sum_pixels = 0.0
        num_elements = 0
        for i in range(len(self.images_paths)):
            img_name = self.images_paths[i]
            image_path = os.path.join(self.images_folder, img_name)
            image = read_image(image_path).float()
            sum_pixels += image.sum()
            num_elements += torch.numel(image)

        self.mean = sum_pixels / num_elements

        std = torch.tensor(0.0)
        for i in range(len(self.images_paths)):
            img_name = self.images_paths[i]
            image_path = os.path.join(self.images_folder, img_name)
            image = read_image(image_path).float()
            std += ((image - self.mean) ** 2).sum()

        self.std = torch.sqrt(std / (num_elements - 1))

    def split_dataset(self, train_ratio: float, val_ratio: float) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
        """
        Splits the dataset into train validation and test.
        The classes are equally represented in each split.
        :param train_ratio: Ratio of train samples (between 0 and 1).
        :param val_ratio: Ratio of validation samples (between 0 and 1).
        :return: The train, validation and test subsets.
        """
        assert train_ratio + val_ratio <= 1, "Train ratio and val ratio must be less than or equal to 1"

        test_ratio = 1 - train_ratio - val_ratio

        # We want each class to be equally represented in each split, given the classes
        train_idx = []
        val_idx = []
        test_idx = []

        # Keep track of the number of images representing each class in each split
        train_representation = np.zeros(len(configs.globals.CLASSES))
        val_representation = np.zeros(len(configs.globals.CLASSES))
        test_representation = np.zeros(len(configs.globals.CLASSES))

        for i in range(len(self)):
            semantic_annotation = self[i]['semantic']

            class_in_image = torch.unique(semantic_annotation)
            # Make it an integer list, and replace the 255 class with -1
            class_in_image = [int(x.item()) for x in class_in_image]
            class_in_image = [x if x != 255 else -1 for x in class_in_image]

            # For each class add a vote if the class is under-represented in the split
            train_votes = 0
            val_votes = 0
            test_votes = 0

            for class_ in class_in_image:
                # Use ratios to take into account the final number of images in each split
                train_value = train_representation[class_] / train_ratio,
                val_value = val_representation[class_] / val_ratio
                test_value = test_representation[class_] / test_ratio
                # Add a vote to the split with the least representation of the class, if all equal add to train
                if train_value <= val_value and train_value <= test_value:
                    train_votes += 1
                elif val_value < train_value and val_value <= test_value:
                    val_votes += 1
                elif test_value < train_value and test_value < val_value:
                    test_votes += 1

            # Add the image to the split with the most votes
            if train_votes >= val_votes and train_votes >= test_votes:
                train_idx.append(i)
                for class_ in class_in_image:
                    train_representation[class_] += 1
            elif val_votes > train_votes and val_votes >= test_votes:
                val_idx.append(i)
                for class_ in class_in_image:
                    val_representation[class_] += 1
            elif test_votes > train_votes and test_votes > val_votes:
                test_idx.append(i)
                for class_ in class_in_image:
                    test_representation[class_] += 1

        # Keep track of the indices for each split
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        train_dataset = torch.utils.data.Subset(self, train_idx)
        val_dataset = torch.utils.data.Subset(self, val_idx)
        test_dataset = torch.utils.data.Subset(self, test_idx)

        return train_dataset, val_dataset, test_dataset


def fetch_data_from_wandb():
    """
    Get the dataset from the last wandb artifact.
    :return: The path to the dataset.
    """

    artifact = wandb.use_artifact(wandb_config.DATA_NAME + ":latest", type="dataset-suadd")
    artifact_dir = artifact.download(root=configs.globals.DATASET_PATH)
    return artifact_dir
