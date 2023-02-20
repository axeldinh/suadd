import torch

from configs.globals import CLASSES
from losses import CrossEntropyMSE
from models.unet import UNet
from transforms.transforms_1 import TransformSet1

unet_no_depth_center = {
    ##############################
    # Experiment configuration
    ##############################
    "idx": 0,

    ##############################
    # Data configuration
    ##############################
    "store_images": True,  # If true, all images in the folder will be stored in memory at once
    "number_images": None,  # Number of images contained in the whole dataset if None all images will be used
    "batch_size": 8,
    "num_workers": 0,
    "train_ratio": 0.8,
    "val_ratio": 0.1,

    ##############################
    # Model configuration
    ##############################
    "model_args": {
        "conv_size": 5,
        "channels_in": 1,
        "first_layer_channels": 64,
        "depth": 5,
        "num_classes": len(CLASSES),
        "return_depth": False
    },
    "model": UNet,

    ##############################
    # Transform configuration
    ##############################

    "transform_args": {
        "patch_size": 512,
        "crop_position": "center",
    },
    "transform": TransformSet1,

    ##############################
    # Training configuration
    ##############################

    "loss_args": {},
    "loss": CrossEntropyMSE,
    "weighted_loss": True,
    "optimizer": torch.optim.Adam,
    "scheduler": None,
    "lr": 3e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    ##############################
    # Wandb configuration
    ##############################
    "use_wandb": True,
    "monitor": "validation/loss",
    "monitor_mode": "min",
    "log_every_n_steps": 1,
    "val_check_interval": 1.0
}

configs = {
    0: unet_no_depth_center,
}
