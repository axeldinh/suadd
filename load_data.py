import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
from configs.paths import dataset_path
from utils.utils_image import make_overlay

from skimage.io import imshow, imread
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import wandb
import configs.wandb as wandb_config


class ImageDataset(Dataset):

    def __init__(self, folder, transform=None, subset=False):
        self.images_folder = os.path.join(folder, 'inputs')
        self.semantic_annotations_folder = os.path.join(folder, 'semantic_annotations')
        self.depth_annotations_folder = os.path.join(folder, 'depth_annotations')
        self.images_paths = os.listdir(self.images_folder)
        self.semantic_annotations_paths = os.listdir(self.semantic_annotations_folder)
        self.depth_annotations_paths = os.listdir(self.depth_annotations_folder)
        if subset:
            self.images_paths = self.images_paths[:10]
            self.semantic_annotations_paths = self.semantic_annotations_paths[:1]
            self.depth_annotations_paths = self.depth_annotations_paths[:1]
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.images_paths[idx])
        semantic_annotation_path = os.path.join(self.semantic_annotations_folder, self.semantic_annotations_paths[idx])
        depth_annotation_path = os.path.join(self.depth_annotations_folder, self.depth_annotations_paths[idx])

        image = read_image(image_path)[0].float()
        semantic_annotation = read_image(semantic_annotation_path)[0].float()
        depth_annotation = imread(depth_annotation_path).astype(np.float32)
        depth_annotation = torch.from_numpy(depth_annotation)

        if self.transform:
            image = self.transform(image)

        return image, semantic_annotation, depth_annotation

def create_table(dataset):

    table = wandb.Table(columns=["File_Name", "Part_1", "Part_2", "Image", "Depth", "Dataset"] + list(wandb_config.CLASSES.values()))

    images_paths = dataset.images_paths

    for i in trange(len(dataset)):
        img_name = images_paths[i]
        img, sem, dep = dataset[i]
        class_id_in_img = torch.unique(sem)
        class_in_img = [wandb_config.CLASSES[c.item()] for c in class_id_in_img]

        table.add_data(
            img_name,
            img_name.split("-")[0],
            img_name.split("-")[1],
            wandb.Image(
                img.float(),
                masks={
                    "predictions": {
                        "mask_data": sem.numpy(),
                        "class_labels": wandb_config.CLASSES,
                    }
                }
            ),
            wandb.Image(
                dep,
                masks={
                    "predictions": {
                        "mask_data": sem.numpy(),
                        "class_labels": wandb_config.CLASSES,
                    }
                }
            ),
            wandb_config.RAW_DATA,
            *[int(c in class_in_img) for c in wandb_config.CLASSES.values()]
        )

    return table

def main():
    dataset = ImageDataset(dataset_path, subset=True)
    run = wandb.init(
        project=wandb_config.WANDB_PROJECT, 
        entity=wandb_config.ENTITY,
        name="upload_data",
        job_type="upload-data"
        )
    artifact = wandb.Artifact(wandb_config.RAW_DATA, type="raw_data")
    artifact.add_dir(os.path.join(dataset_path, 'inputs'), name="images")
    artifact.add_dir(os.path.join(dataset_path, 'semantic_annotations'), name="semantics")
    artifact.add_dir(os.path.join(dataset_path, 'depth_annotations'), name="depths")
    table = create_table(dataset)
    artifact.add(table, "suadd_table")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()



