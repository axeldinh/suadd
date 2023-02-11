import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Dice

from configs.globals import DATASET_PATH, CLASSES
from utils.datasets import ImageDataset, fetch_data_from_wandb
from utils.metrics import si_log as SILOG
from utils.utils_image import unpatchify


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
        self.si_log = SILOG

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
        self.dataset_path = DATASET_PATH

        # Set the seed for pytorch lightning, torch, numpy python.random
        self.seed = config["seed"]
        pl.seed_everything(self.seed)

        # Load the datasets
        self.get_dataset()

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
        return self.evaluation_step(batch, batch_idx, mode='val')

    def validation_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, mode='val')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, mode='test')

    def test_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, mode='test')

    def evaluation_step(self, batch, batch_idx, mode):
        # Here the batch has shape (size_batch, num_patches, height, width)
        losses = []
        semantic_losses = []
        depth_losses = []
        ious = []
        si_logs = []
        for i in range(batch['image'].shape[0]):
            image_shape = batch['image_shape'][i]
            patches = batch['image'][i]
            semantic = batch['semantic'][i]
            depth = batch['depth'][i].squeeze(1)
            output = self(patches)
            semantic_out = output['semantic']
            semantic_out = unpatchify(semantic_out, image_shape).unsqueeze(0)
            depth_out = output['depth']
            if depth_out is not None:
                depth_out = unpatchify(depth_out, image_shape).squeeze(1)
            else:
                depth_out = None
            semantic_pred = torch.argmax(semantic_out, dim=1)
            depth[depth == 0] = torch.nan
            losses_img = self.loss(semantic_out, depth_out, semantic, depth)
            iou = self.iou(semantic_pred, semantic)
            if depth_out is not None:
                si_log = self.si_log(depth_out.flatten(), depth.flatten())
            else:
                si_log = 0
            losses.append(losses_img['loss'])
            semantic_losses.append(losses_img['semantic_loss'])
            depth_losses.append(losses_img['depth_loss'])
            ious.append(iou)
            si_logs.append(si_log)
        metrics = {
            'loss': torch.stack(losses).mean(),
            'semantic_loss': torch.stack(semantic_losses).sum(),
            'depth_loss': torch.stack(depth_losses).sum(),
            'iou': torch.stack(ious).sum(),
            'si_log': torch.stack(si_logs).sum()
        }
        return metrics

    def evaluation_epoch_end(self, outputs, mode):

        num_images = len(self.val_set) if mode == 'val' else len(self.test_set)

        all_metrics = {}
        for metric in outputs[0].keys():
            all_metrics[metric] = 0
        for output in outputs:
            for metric in output.keys():
                all_metrics[metric] += output[metric]
        for metric in all_metrics.keys():
            all_metrics[metric] /= num_images

        for metric in all_metrics.keys():
            self.log(f'{mode}_{metric}', all_metrics[metric])

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

    # Batch size is 1 for validation and test
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)
