import os

import numpy as np
import torch
import wandb
from skimage.io import imread
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image

import configs.wandb as wandb_config


class ImageDataset(Dataset):
    """
    Dataset for semantic segmentation and depth estimation for the SUADD Challenge.
    The transforms are applied at each iteration, so they should be random.
    """

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
        semantic_annotation[semantic_annotation == 255] = len(wandb_config.CLASSES) - 1
        depth_annotation = imread(depth_annotation_path).astype(np.float32)
        depth_annotation = torch.from_numpy(depth_annotation).unsqueeze(0)

        if self.transform:
            image, semantic_annotation, depth_annotation = self.transform(image, semantic_annotation, depth_annotation)

        return image, semantic_annotation, depth_annotation


def fetch_data_from_wandb():
    """
    Get the dataset from the last wandb artifact.
    :return: The path to the dataset.
    """
    run = wandb.init(
        project=wandb_config.WANDB_PROJECT,
        entity=wandb_config.ENTITY,
        name="download_data",
        job_type="download-data"
    )
    artifact = run.use_artifact(wandb_config.DATA_NAME + ":latest", type="dataset-suadd")
    artifact_dir = artifact.download()
    return artifact_dir


def split_dataset(dataset: Dataset, train_ratio: float, val_ratio: float) -> tuple([Subset, Subset, Subset]):
    """
    Splits the dataset into train validation and test.
    The classes are equally represented in each split.
    :param dataset: Dataset to split.
    :param train_ratio: Ratio of train samples (between 0 and 1).
    :param val_ratio: Ratio of validation samples (between 0 and 1).
    :return: The train, validation and test subsets.
    """
    assert train_ratio + val_ratio <= 1, "Train ratio and val ratio must be less than or equal to 1"

    test_ratio = 1 - train_ratio - val_ratio

    # If want each class to be equally represented in each split, given the classes
    train_idx = []
    val_idx = []
    test_idx = []

    # Keep track of the number of images representing each class in each split
    train_representation = np.zeros(len(wandb_config.CLASSES))
    val_representation = np.zeros(len(wandb_config.CLASSES))
    test_representation = np.zeros(len(wandb_config.CLASSES))

    for i in range(len(dataset)):
        _, semantic_annotation, _ = dataset[i]

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

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset
