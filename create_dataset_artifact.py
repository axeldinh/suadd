import os

import torch
import wandb
from tqdm import trange

import configs.globals
import configs.wandb as wandb_config
from configs.globals import DATASET_PATH
from utils.datasets import ImageDataset


def create_table(dataset):
    table = wandb.Table(
        columns=["File_Name", "Part_1", "Part_2", "Image", "Depth", "Dataset"] + list(configs.globals.CLASSES.values()))

    images_paths = dataset.images_paths

    for i in trange(len(dataset)):
        img_name = images_paths[i]
        img = dataset[i]['image']
        sem = dataset[i]['semantic']
        dep = dataset[i]['depth']
        class_id_in_img = torch.unique(sem)
        class_in_img = [configs.globals.CLASSES[c.item()] for c in class_id_in_img]

        table.add_data(
            img_name,
            img_name.split("-")[0],
            img_name.split("-")[1],
            wandb.Image(
                img.float(),
                masks={
                    "predictions": {
                        "mask_data": sem.numpy()[0],
                        "class_labels": configs.globals.CLASSES,
                    }
                }
            ),
            wandb.Image(
                dep,
                masks={
                    "predictions": {
                        "mask_data": sem.numpy()[0],
                        "class_labels": configs.globals.CLASSES,
                    }
                }
            ),
            wandb_config.RAW_DATA,
            *[int(c in class_in_img) for c in configs.globals.CLASSES.values()]
        )

    return table


def main():
    dataset = ImageDataset(DATASET_PATH, size=None)
    run = wandb.init(
        project=wandb_config.WANDB_PROJECT,
        entity=wandb_config.ENTITY,
        name="upload_data",
        job_type="upload-data"
    )
    artifact = wandb.Artifact(wandb_config.DATA_NAME, type="dataset-suadd")
    artifact.add_dir(os.path.join(DATASET_PATH, 'inputs'), name="inputs")
    artifact.add_dir(os.path.join(DATASET_PATH, 'semantic_annotations'), name="semantic_annotations")
    artifact.add_dir(os.path.join(DATASET_PATH, 'depth_annotations'), name="depth_annotations")
    table = create_table(dataset)
    artifact.add(table, "suadd_table")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
