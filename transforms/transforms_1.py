import torch
import torchvision.transforms as T

from utils.transforms import RandomGamma, RandomNoise


class TransformSet1:
    """
    First of transforms sets.
    Applies:
        - Random crop in the image
        - Random Blur
        - Random Noise
        - Random Horizontal Flip
        - Random Vertical Flip
        - Random Brightness, Contrast, Saturation, Hue
        - Random Gamma
        - Random Affine
        - Random Perspective
    """

    def __init__(self, p=0.5, patch_size=256):

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
            T.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10,
                           interpolation=T.InterpolationMode.NEAREST),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=T.InterpolationMode.NEAREST),
        ])

    def __call__(self, image, semantic, depth, is_mask=False):
        """
        Applies the transforms to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :param is_mask: if True, only applies horizontal, vertical flips, affine and perspective transforms
        :return: transformed torch gray images (1, height, width)
        """

        if not is_mask and image.max() > 1.0:
            image = image / 255.

        image = self.value_transforms(image)

        image, semantic, depth = self.shape_transforms(torch.stack([image, semantic, depth]))


        return image, semantic, depth
