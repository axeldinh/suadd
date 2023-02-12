import os

import numpy as np
import torch
import wandb
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import trange, tqdm

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

        self.mean = None
        self.std = None

        self.classes_count = None

        self.size = size
        self.store_images = store_images

        self.dataset_path = folder
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
            image = self.images[idx]
            semantic_annotation = self.semantic_annotations[idx]
            depth_annotation = self.depth_annotations[idx]
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

        self.classes_count = {}
        self.classes_ratio = {}

        for i in tqdm(self.train_idx, leave=None):
            if self.store_images:
                image = self.images[i]
                semantic = self.semantic_annotations[i]
            else:
                img_name = self.images_paths[i]
                image_path = os.path.join(self.images_folder, img_name)
                image = read_image(image_path).float()
                semantic_path = os.path.join(self.semantic_annotations_folder, img_name)
                semantic = read_image(semantic_path).float()
                semantic[semantic == 255] = len(configs.globals.CLASSES) - 1
            sum_pixels += image.sum()
            num_elements += torch.numel(image)
            for class_ in configs.globals.CLASSES.keys():
                if class_ not in self.classes_count:
                    self.classes_count[class_] = 0
                self.classes_count[class_] += (semantic == class_).float().sum().item()

        for class_ in self.classes_count.keys():
            self.classes_ratio[class_] = self.classes_count[class_] / num_elements


        self.mean = sum_pixels / num_elements

        std = torch.tensor(0.0)
        for i in tqdm(self.train_idx, leave=None):
            if self.store_images:
                image = self.images[i]
            else:
                img_name = self.images_paths[i]
                image_path = os.path.join(self.images_folder, img_name)
                image = read_image(image_path).float()
            std += ((image - self.mean) ** 2).sum()

        self.std = torch.sqrt(std / (num_elements - 1))

    def get_weights(self):
        """
        Get the weights for each class
        :return: the weights for each class
        """
        weights = torch.zeros(len(configs.globals.CLASSES))
        for class_ in configs.globals.CLASSES.keys():
            if self.classes_ratio[class_] == 0:
                weights[class_] = 0
            else:
                weights[class_] = 1 / self.classes_ratio[class_]
        weights /= weights.sum()
        return weights

    def split_dataset(self, train_ratio: float, val_ratio: float):
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

    def prepare_dataset(self, train_ratio: float, val_ratio: float):
        """
        Prepares the dataset. If store_images is set to True, the images are loaded in memory.
        Then the dataset is split into train, validation and test.
        Finally, the mean and std of the training dataset are computed.
        :param train_ratio: float between 0 and 1
        :param val_ratio: float between 0 and 1
        :return:
        """

        if self.store_images:
            print("ImageDataset.store_images set to True, loading images in memory...")
            self.images = []
            self.semantic_annotations = []
            self.depth_annotations = []
            for i in trange(len(self.images_paths)):
                img_name = self.images_paths[i]
                depth_annotation, image, semantic_annotation = self.load_datapoint(img_name)
                self.images.append(image)
                self.semantic_annotations.append(semantic_annotation)
                self.depth_annotations.append(depth_annotation)

        print("Splitting dataset into train, validation and test...")
        self.train_set, self.val_set, self.test_set = self.split_dataset(train_ratio, val_ratio)

        if self.transform is not None:
            print("Computing mean and std of the training dataset...")
            self.compute_mean_std()
            self.transform.set_mean_std(self.mean, self.std)

    def get_data_splits(self):
        """
        Returns the train, validation and test subsets.
        :return: The train, validation and test subsets.
        """
        if self.train_set is None or self.val_set is None or self.test_set is None:
            raise Exception("The dataset has not been prepared yet. Call prepare_dataset() first.")
        return self.train_set, self.val_set, self.test_set



def fetch_data_from_wandb():
    """
    Get the dataset from the last wandb artifact.
    :return: The path to the dataset.
    """

    artifact = wandb.use_artifact(wandb_config.DATA_NAME + ":latest", type="dataset-suadd")
    artifact_dir = artifact.download(root=configs.globals.DATASET_PATH)
    return artifact_dir
