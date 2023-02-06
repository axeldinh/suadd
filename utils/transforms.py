import torch
from torch import nn as nn
from torchvision import transforms as T


class RandomGamma(nn.Module):
    """
    Randomly changes the gamma of a batch of torch gray images.
    """

    def __init__(self, gamma_range=(0.5, 1.5)):
        """
        :param gamma_range: range of gamma values
        """

        super(RandomGamma, self).__init__()

        if not isinstance(gamma_range, tuple) and not isinstance(gamma_range, list):
            if isinstance(gamma_range, int) or isinstance(gamma_range, float):
                gamma_range = (1 - gamma_range, 1 + gamma_range)

        self.gamma_range = gamma_range

    def forward(self, image):
        """
        Applies the transform to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :return: transformed torch gray images (1, height, width)
        """

        gamma = torch.rand(1) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]

        return T.functional.adjust_gamma(image, gamma.item())


class RandomNoise(nn.Module):

    def __init__(self, noise_range=(0, 0.5)):
        """
        :param noise_range: range of noise values
        """

        super(RandomNoise, self).__init__()

        if not isinstance(noise_range, tuple) and not isinstance(noise_range, list):
            if isinstance(noise_range, int) or isinstance(noise_range, float):
                noise_range = (0, noise_range)

        self.noise_range = noise_range

    def forward(self, image):
        """
        Applies the transform to a batch of torch gray images.
        :param image: torch gray images (1, height, width)
        :return: transformed torch gray images (1, height, width)
        """

        noise = torch.rand(1) * (self.noise_range[1] - self.noise_range[0]) + self.noise_range[0]

        return image + torch.randn(image.shape) * noise.item()
