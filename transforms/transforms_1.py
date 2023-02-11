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

    def __init__(self, p=0.5, patch_size=256):

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
            T.RandomCrop(size=(patch_size, patch_size)),
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
        ])

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, image):

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

        # Normalize
        image = self.normalize(image)

        return image, semantic, depth

    @staticmethod
    def eval_untransform(image, semantic, depth, original_shape):
        """
        Inverse transforms the image to its original size, for evaluation images.
        :param image: torch gray images (1, height, width) or several patches (1, patch_size, patch_size)
        :param semantic: torch gray images (1, height, width) or several patches (1, patch_size, patch_size)
        :param depth: torch gray images (1, height, width) or several patches (1, patch_size, patch_size)
        :param original_shape: original shape of the image
        :return: untransformed torch gray images (1, height, width)
        """

        dims = len(image.shape)
        if dims == 3:
            num_patches = image.shape[0]
            patch_size = image.shape[1]
        elif dims == 4:
            if image.shape[1] != 1:
                raise Exception(
                    "Image must be gray. It should have shape (num_patches, 1, patch_size, patch_size)")
            num_patches = image.shape[1]
            patch_size = image.shape[2]
        else:
            raise Exception("Image must have 3 or 4 dimensions.")

        image = image.view(num_patches, 1, patch_size, patch_size)

        image = unpatchify(image, original_shape)

        image = image.reshape(semantic.shape)

        return image, semantic, depth
