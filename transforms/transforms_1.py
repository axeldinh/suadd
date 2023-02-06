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

    def __call__(self, image):
        """
        Applies the transforms to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :return: transformed torch gray images (1, height, width)
        """

        image = image / 255.

        return self.transforms(image)
