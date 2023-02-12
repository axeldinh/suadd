import os

import pytorch_lightning as pl
import torch
import wandb
from skimage.io import imsave
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Dice

from configs.globals import DATASET_PATH, CLASSES
from utils.datasets import ImageDataset, fetch_data_from_wandb
from utils.metrics import compute_depth_metrics
from utils.utils_image import make_overlay


class LitModel(pl.LightningModule):

    def __init__(self, config):
        # Save the git commit hash
        super().__init__()

        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.config = config

        self.model = config["model"](**config["model_args"])
        self.loss = config["loss"]
        self.optimizer = config["optimizer"]
        self.scheduler = config["scheduler"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.epochs = config["epochs"]
        self.transform = config["transform"](**config["transform_args"])
        self.train_ratio = config["train_ratio"]
        self.val_ratio = config["val_ratio"]
        self.dataset_path = DATASET_PATH
        self.save_path = config["save_path"]
        os.makedirs(self.save_path, exist_ok=True)

        # Set the seed for pytorch lightning, torch, numpy python.random
        self.seed = config["seed"]
        pl.seed_everything(self.seed)

        # Load the datasets
        self.get_dataset()

        self.example_input_array = torch.rand(1, 1, config["transform_args"]["patch_size"],
                                              config["transform_args"]["patch_size"])

        # Metrics for semantic segmentation and depth estimation
        self.SEMANTIC_METRICS = {
            'semantic/iou': JaccardIndex(task='multiclass', num_classes=len(CLASSES), average=None,
                                         ignore_index=len(CLASSES) - 1).to(config["device"]),
            'semantic/dice': Dice(num_classes=len(CLASSES), average='macro',
                                  ignore_index=len(CLASSES) - 1).to(config["device"]),
        }

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        semantic = batch['semantic']
        depth = batch['depth']

        semantic_out, depth_out = self(images)
        losses = self.loss(semantic_out, depth_out, semantic, depth)
        self.log('train/loss', losses['loss'])
        self.log('train/semantic_loss', losses['semantic_loss'])
        self.log('train/depth_loss', losses['depth_loss'])
        return losses

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, mode='validation')

    def validation_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, mode='validation')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, mode='test')

    def test_epoch_end(self, outputs):
        self.evaluation_epoch_end(outputs, mode='test')

    def evaluation_step(self, batch, batch_idx, mode):
        # Here the batch has shape (size_batch, num_patches, height, width)

        all_metrics = {}
        for i in range(batch['image'].shape[0]):

            image_shape = batch['image_shape'][i]
            patches = batch['image'][i]
            semantic = batch['semantic'][i]
            depth = batch['depth'][i].squeeze(1)

            ##############################
            # Get the predictions for each image
            ##############################

            semantic_out, depth_out = self(patches)
            semantic_out = self.transform.eval_untransform(semantic_out, image_shape)
            if depth_out is not None:
                depth_out = self.transform.eval_untransform(depth_out, image_shape)

            ##############################
            # Compute the metrics
            ##############################

            # Make sure everything has the proper shape
            semantic = semantic.view(1, semantic.shape[-2], semantic.shape[-1])
            depth = depth.view(depth.shape[-2], depth.shape[-1])
            semantic_out = semantic_out.view(1, len(CLASSES.keys()), semantic_out.shape[-2], semantic_out.shape[-1])
            if depth_out is not None:
                depth_out = depth_out.view(depth_out.shape[-2], depth_out.shape[-1])
            semantic_pred = torch.argmax(semantic_out, dim=1)

            depth[depth == 0] = torch.nan
            losses_img = self.loss(semantic_out, depth_out, semantic, depth)
            semantic_metrics = self.compute_semantic_metrics(semantic_pred, semantic)
            all_metrics_img = {**losses_img, **semantic_metrics}
            depth_metrics = None
            shape = (depth.shape[-2], depth.shape[-1])
            if depth_out is not None:
                depth_metrics = self.compute_depth_metrics(depth_out.reshape(shape), depth.reshape(shape),
                                                           semantic.reshape(shape))
                all_metrics_img.update(depth_metrics)

            ##############################
            # Store them
            ##############################

            for k, v in all_metrics_img.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

            #########################################################
            # If test mode, save the predictions locally and on wandb
            #########################################################

            if mode == 'test':
                image = self.transform.eval_untransform(patches, image_shape)
                image.view(1, image.shape[-2], image.shape[-1])
                save_path = os.path.join(self.save_path, 'test')
                os.makedirs(save_path, exist_ok=True)
                self.save_predictions_images(image, semantic, depth, semantic_pred, depth_out,
                                             image_name=batch['image_name'][i], save_path=save_path)

        ##############################
        # Compute the sum, it will be averaged later
        ##############################

        for k, v in all_metrics.items():
            all_metrics[k] = torch.stack(v).sum()

        return all_metrics

    def evaluation_epoch_end(self, outputs, mode):
        """
        Average the metrics over the whole dataset and log the results
        """

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
            # Log the metrics, adding '/' to  get different groups in wandb
            self.log(f'{mode}/{metric.replace("semantic_", "semantic/").replace("depth_", "depth/")}',
                     all_metrics[metric])

    def compute_semantic_metrics(self, prediction, target):
        metrics = {}
        for metric_name, metric in self.SEMANTIC_METRICS.items():
            if metric_name == "semantic/iou":
                ious = metric(prediction, target.long())
                for i in CLASSES.keys():
                    metrics[f"semantic/ious/{CLASSES[int(i)]}"] = ious[i]
                # Unknown class is not included in the iou
                metrics["semantic/ious/mean"] = ious[:-1].mean()
            metrics[metric_name] = metric(prediction, target.long())
        return metrics

    def compute_depth_metrics(self, prediction, target, semantic):
        """
        Compute the depth metrics for each class and for all the image
        :param prediction: depth prediction
        :param target: depth target
        :param semantic: semantic target
        :return:
        """
        metrics = {}
        zeros_classes = []
        for class_ in CLASSES.keys():
            mask = semantic == class_
            if mask.sum() == 0:
                zeros_classes.append(class_)
            else:
                depth_pred = prediction[mask]
                depth_target = target[mask]
                depth_metrics = compute_depth_metrics(depth_pred.flatten(), depth_target.flatten())
                for k, v in depth_metrics.items():
                    metrics[f"depth/{k}/{CLASSES[class_]}"] = v
        for class_ in zeros_classes:
            for k in depth_metrics.keys():
                metrics[f"depth/{k}/{CLASSES[class_]}"] = torch.tensor(0.).to(prediction.device)

        metrics_full = compute_depth_metrics(prediction.flatten(), target.flatten())
        metrics_full = {f"depth/{k}/full": v for k, v in metrics_full.items()}
        metrics.update(metrics_full)

        return metrics

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def get_dataset(self):
        if not os.path.exists(self.dataset_path) or self.config["use_wandb"]:
            self.dataset_path = fetch_data_from_wandb()
        dataset = ImageDataset(
            self.dataset_path,
            transform=self.transform,
            size=self.config["number_images"],
            store_images=self.config["store_images"]
        )
        self.train_set, self.val_set, self.test_set = dataset.split_dataset(self.train_ratio, self.val_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    # Batch size is 1 for validation and test
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)

    def save_predictions_images(self, image, semantic_target, depth_target,
                                semantic_pred, depth_out, image_name, save_path):
        """
        Save the predictions locally and on wandb
        :param image: image of shape (1, height, width), torch tensor
        :param semantic_target: semantic target of shape (1, height, width), torch tensor
        :param depth_target: depth target of shape (height, width), torch tensor
        :param semantic_pred: semantic prediction of shape (1, height, width), torch tensor
        :param depth_out: depth prediction of shape (height, width), torch tensor
        :param image_name: name of the image
        :param save_path: path to save the images
        :return: None
        """

        # Everything to cpu
        image = image.cpu()
        semantic_target = semantic_target.cpu()
        depth_target = depth_target.cpu()
        semantic_pred = semantic_pred.cpu()
        depth_out = depth_out.cpu()

        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).to(torch.uint8)

        # Save the prediction overlay locally
        overlay = make_overlay(image, semantic_pred)
        imsave(os.path.join(save_path, f'{image_name}_overlay.png'), overlay.numpy().transpose(1, 2, 0))

        # Save the prediction overlay on wandb
        if self.config["use_wandb"]:
            wandb_image = wandb.Image(image.float(), masks={
                "predictions": {
                    "mask_data": semantic_pred[0].to(torch.uint8).numpy(),
                    "class_labels": CLASSES
                },
                "ground truth": {
                    "mask_data": semantic_target[0].to(torch.uint8).numpy(),
                    "class_labels": CLASSES
                }
            })
            wandb.log({f"test/semantic_overlays/{image_name}": wandb_image})
        else:
            # Save the overlay to tensorboard
            self.logger.experiment.add_image(f"test/semantic_overlays/{image_name}", overlay, self.current_epoch)

        if depth_out is not None:
            mask = ~torch.isnan(depth_target)
            depth_error = torch.zeros_like(depth_target)
            depth_error[mask] = torch.abs(depth_out[mask] - depth_target[mask])
            depth_error = torch.clamp(depth_error, 0, 255)
            depth_error = depth_error.to(torch.uint8)
            # Save the depth error locally
            imsave(os.path.join(save_path, f'{image_name}_depth_error.png'), depth_error.numpy(), check_contrast=False)
            # Save the depth error on wandb
            if self.config["use_wandb"]:
                wandb.log({f"test/depth_error_images/{image_name}": wandb.Image(depth_error.float())})
            else:
                # Save the depth error to tensorboard
                self.logger.experiment.add_image(f"test/depth_error_images/{image_name}", depth_error.unsqueeze(0),
                                                 self.current_epoch)
            depth_uint8 = torch.clamp(depth_out, 0, 255).to(torch.uint8)
            # Save the depth prediction locally
            imsave(os.path.join(save_path, f'{image_name}_depth.png'), depth_uint8.numpy(), check_contrast=False)
            # Save the depth prediction on wandb
            if self.config["use_wandb"]:
                wandb.log({f"test/depth_images/{image_name}": wandb.Image(depth_uint8.float())})
            else:
                # Save the depth prediction to tensorboard
                self.logger.experiment.add_image(f"test/depth_images/{image_name}", depth_uint8.unsqueeze(0),
                                                 self.current_epoch)
