import torch
import torchvision.transforms as T

from utils.transforms import RandomGamma, RandomNoise
from utils.utils_image import patchify, unpatchify


class TransformSet1:
    """
    First of transforms sets.
    Applies:
        - Random Blur
        - Random Noise
        - Color Jitter
        - Random Gamma
        - Random Crop
        - Random Horizontal Flip
        - Random Vertical Flip
        - Random Affine
        - Random Perspective
        - Normalize
    """

    def __init__(self, p=0.5, patch_size=256, crop_position="center"):

        if crop_position == "center":
            crop = T.CenterCrop(size=(patch_size, patch_size))
        elif crop_position == "random":
            crop = T.RandomCrop(size=(patch_size, patch_size))
        else:
            raise ValueError("crop_position must be 'center' or 'random'.")

        self.mean = None
        self.std = None
        self.patch_size = patch_size
        self.value_transforms = T.Compose([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            RandomNoise(0.05),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            RandomGamma(0.3),
        ])

        self.shape_transforms = T.Compose([
            crop,
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
        ])

    def set_mean_std(self, mean, std):
        """
        Set the mean and std for normalization.
        :param mean: Mean of training set, based on [0, 255] images
        :param std: Std of training set, based on [0, 255] images
        :return:
        """
        self.mean = mean / 255.
        self.std = std / 255.

    def normalize(self, image):
        """
        Normalize the image.
        :param image:
        :return:
        """
        if self.mean is None or self.std is None:
            raise Exception("Mean and Std must be set before normalization.")

        return T.functional.normalize(image, mean=self.mean, std=self.std)

    def __call__(self, image, semantic, depth, train=True):
        """
        Applies the transforms to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :param semantic: torch gray images (1, height, width)
        :param depth: torch gray images (1, height, width)
        :param train: if True, apply the transforms, else normalize and make patches
        :return: transformed torch gray images (1, height, width) or several patches (1, patch_size, patch_size)
        """

        if image.max() > 1.0:
            image = image / 255.

        if train:
            image = self.value_transforms(image)
            image, semantic, depth = self.shape_transforms(torch.stack([image, semantic, depth]))

        else:
            image = patchify(image, self.patch_size)

        # Normalize (mean and std are based on uint8 images)
        image = self.normalize(image)

        return image, semantic, depth

    @staticmethod
    def eval_untransform(image, original_shape):
        """
        Inverse transforms the image to its original size, for evaluation images.
        :param image: torch gray images (num_patches, C, patch_size, patch_size)
        :param original_shape: original shape of the image
        :return: untransformed torch gray images (C, height, width)
        """

        dims = len(image.shape)
        if dims == 3:
            num_patches = image.shape[0]
            patch_size = image.shape[-1]
            num_channels = 1
        elif dims == 4:
            num_patches = image.shape[0]
            patch_size = image.shape[-1]
            num_channels = image.shape[1]
        else:
            raise Exception("Image must have 3 or 4 dimensions.")

        image = image.view(num_patches, num_channels, patch_size, patch_size)

        image = unpatchify(image, original_shape)

        return image
