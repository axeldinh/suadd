import os
from tqdm import trange
import wandb
import torch

from configs.paths import dataset_path
import configs.wandb as wandb_config
from configs.globals import *
from utils.datasets import ImageDataset

def create_table(dataset):

    table = wandb.Table(columns=["File_Name", "Part_1", "Part_2", "Image", "Depth", "Dataset"] + list(wandb_config.CLASSES.values()))

    images_paths = dataset.images_paths

    for i in trange(len(dataset)):
        img_name = images_paths[i]
        img, sem, dep = dataset[i]
        class_id_in_img = torch.unique(sem)
        class_in_img = [wandb_config.CLASSES[c.item()] for c in class_id_in_img]

        # Check that the classes are well represented

        #print(wandb_config.CLASSES)
        #print([int(c in class_in_img) for c in wandb_config.CLASSES.values()])
        #for cls in wandb_config.CLASSES:
        #    sem_temp = sem.clone()
        #    sem_temp[sem!=cls] = 0
        #    overlay = make_overlay(img.type(torch.uint8), sem_temp)
        #    imshow(overlay.numpy().transpose(1, 2, 0))
        #    plt.title(wandb_config.CLASSES[cls])
        #    plt.show()

        table.add_data(
            img_name,
            img_name.split("-")[0],
            img_name.split("-")[1],
            wandb.Image(
                img.float(),
                masks={
                    "predictions": {
                        "mask_data": sem.numpy()[0],
                        "class_labels": wandb_config.CLASSES,
                    }
                }
            ),
            wandb.Image(
                dep,
                masks={
                    "predictions": {
                        "mask_data": sem.numpy()[0],
                        "class_labels": wandb_config.CLASSES,
                    }
                }
            ),
            wandb_config.RAW_DATA,
            *[int(c in class_in_img) for c in wandb_config.CLASSES.values()]
        )

    return table

def main():

    dataset = ImageDataset(dataset_path, size=None)
    run = wandb.init(
        project=wandb_config.WANDB_PROJECT, 
        entity=wandb_config.ENTITY,
        name="upload_data",
        job_type="upload-data"
        )
    artifact = wandb.Artifact(wandb_config.DATA_NAME, type="dataset-suadd")
    artifact.add_dir(os.path.join(dataset_path, 'inputs'), name="images")
    artifact.add_dir(os.path.join(dataset_path, 'semantic_annotations'), name="semantics")
    artifact.add_dir(os.path.join(dataset_path, 'depth_annotations'), name="depths")
    table = create_table(dataset)
    artifact.add(table, "suadd_table")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()



