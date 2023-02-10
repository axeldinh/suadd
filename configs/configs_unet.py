import torch
import torch.nn as nn

from configs.globals import CLASSES
from models.unet import UNet
from transforms.transforms_1 import TransformSet1
from losses.cross_entropy_mse import CrossEntropyMSE

config_1 = {
    "model": UNet(1, 1, 2, num_classes=len(CLASSES), return_depth=True),
    "loss": CrossEntropyMSE(),
    "optimizer": torch.optim.Adam,
    "scheduler": None,
    "lr": 1e-4,
    "batch_size": 4,
    "num_workers": 4,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "transform": TransformSet1(),
    "epochs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0,
}

configs = {
    1: config_1,
}