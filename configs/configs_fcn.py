import os

import torch

from configs.globals import OUTPUTS_PATH, CLASSES
from losses import CrossEntropyMSE
from models.fcn import FCN
from transforms.transforms_1 import TransformSet1

fcn_50_no_depth = {
    ##############################
    # Experiment configuration
    ##############################
    "idx": 100,
    "name": "Experiment_1",
    "save_path": os.path.join(OUTPUTS_PATH, "Experiment_1"),

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
        "resnet": "50",
        "num_classes": len(CLASSES),
        "return_depth": False
    },
    "model": FCN,

    ##############################
    # Transform configuration
    ##############################

    "transform_args": {
        "patch_size": 512,
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
    100: fcn_50_no_depth,
}