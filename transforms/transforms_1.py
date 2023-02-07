import torchvision.transforms as T

from utils.transforms import RandomGamma, RandomNoise


class TransformSet1:
    """
    First of transforms sets.
    Applies:
        - Random Blur
        - Random Noise
        - Random Horizontal Flip
        - Random Vertical Flip
        - Random Brightness, Contrast, Saturation, Hue
        - Random Gamma
        - Random Affine
        - Random Perspective
        - To RGB (3 channels)
    """

    def __init__(self, p=0.5):
        self.transforms = T.Compose([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            RandomNoise(0.05),
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            RandomGamma(0.3),
            T.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])

        self.transforms_mask = T.Compose([
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
            T.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])

    def __call__(self, image, is_mask=False):
        """
        Applies the transforms to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :param is_mask: if True, only applies horizontal, vertical flips, affine and perspective transforms
        :return: transformed torch gray images (1, height, width)
        """

        if not is_mask and image.max() > 1.0:
            image = image / 255.

        if is_mask:
            new_image = self.transforms_mask(image)
        else:
            new_image = self.transforms(image)

        return new_image
