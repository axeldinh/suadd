import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import JaccardIndex, Dice

from configs.globals import *
from configs.paths import dataset_path
from configs.wandb import CLASSES

from utils.datasets import ImageDataset, fetch_data_from_wandb, split_dataset
from utils.metrics import si_log, abs_rel

class LitModel(pl.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Metrics for semantic segmentation
        self.iou = JaccardIndex(task='multiclass', num_classes=len(CLASSES), average='macro')
        self.dice = Dice(num_classes=len(CLASSES), average='macro')

        # Metrics for depth estimation (use numpy)
        self.si_log = si_log
        self.abs_rel = abs_rel

        self.model = config["model"]
        self.loss = config["loss"]
        self.optimizer = config["optimizer"]
        self.scheduler = config["scheduler"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.epochs = config["epochs"]
        self.transform = config["transform"]
        self.train_ratio = config["train_ratio"]
        self.val_ratio = config["val_ratio"]
        self.dataset_path = config["dataset_path"]

        self.get_dataset()
        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.test_loader = self.test_dataloader()


    def get_dataset(self):

        if not os.path.exists(self.dataset_path):
            self.dataset_path = fetch_data_from_wandb()
        dataset = ImageDataset(self.dataset_path, transform=self.transform)
        train_set, val_set, test_set = split_dataset(dataset, self.train_ratio, self.val_ratio)

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":

    config = {
        "model": None,
        "loss": None,
        "optimizer": None,
        "scheduler": None,
        "lr": 0.001,
        "batch_size": 4,
        "num_workers": 4,
        "epochs": 10,
        "transform": None,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "dataset_path": dataset_path
    }

    model = LitModel(config)