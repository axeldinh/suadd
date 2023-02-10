import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Dice

import configs.paths as paths
from configs.wandb import CLASSES
from models.unet import UNet
from utils.datasets import ImageDataset, fetch_data_from_wandb, split_dataset
from utils.metrics import si_log, abs_rel


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.test_set = None
        self.val_set = None
        self.train_set = None

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
        self.dataset_path = paths.dataset_path

        # Set the seed for pytorch lightning, torch, numpy python.random
        self.seed = config["seed"]
        pl.seed_everything(self.seed)

        # Load the datasets
        self.get_dataset()
        self.train_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.test_loader = self.test_dataloader()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        semantic = batch['semantic']
        depth = batch['depth']
        output = self(images)
        losses = self.loss(output['semantic'], output['depth'], semantic, depth)
        self.log('train_loss', losses['loss'])
        self.log('train_semantic_loss', losses['semantic_loss'])
        self.log('train_depth_loss', losses['depth_loss'])
        return losses

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        semantic = batch['semantic']
        depth = batch['depth']
        output = self(images)
        losses = self.loss(output['semantic'], output['depth'], semantic, depth)
        self.log('val_loss', losses['loss'])
        self.log('val_semantic_loss', losses['semantic_loss'])
        self.log('val_depth_loss', losses['depth_loss'])
        return losses

    def test_step(self, batch, batch_idx):
        images = batch['image']
        semantic = batch['semantic']
        depth = batch['depth']
        output = self(images)
        losses = self.loss(output['semantic'], output['depth'], semantic, depth)
        self.log('test_loss', losses['loss'])
        self.log('test_semantic_loss', losses['semantic_loss'])
        self.log('test_depth_loss', losses['depth_loss'])
        return losses

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def get_dataset(self):
        if not os.path.exists(self.dataset_path):
            self.dataset_path = fetch_data_from_wandb()
        dataset = ImageDataset(self.dataset_path, transform=self.transform)
        self.train_set, self.val_set, self.test_set = dataset.split_dataset(self.train_ratio, self.val_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    import torch
    from transforms.transforms_1 import TransformSet1
    from losses.cross_entropy_mse import CrossEntropyMSE

    config = {
        "model": UNet(1, 1, 2, num_classes=len(CLASSES), return_depth=True),
        "loss": CrossEntropyMSE(),
        "optimizer": torch.optim.Adam,
        "scheduler": None,
        "lr": 0.001,
        "batch_size": 4,
        "num_workers": 4,
        "epochs": 2,
        "transform": TransformSet1(),
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0,
    }

    model = LitModel(config)

    for batch in model.train_loader:
        output = model(batch['image'])
        loss = model.loss(output['semantic'], output['depth'], batch['semantic'], batch['depth'])
        print(loss)
        break

    # Train, gpu if available
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=config["epochs"])
    trainer.fit(model)

    # Test
    trainer.test(model)
