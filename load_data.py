import wandb

import configs.wandb as wandb_config
from configs.globals import DATASET_PATH
from utils.datasets import ImageDataset

if __name__ == "__main__":
    run = wandb.init(
        project=wandb_config.WANDB_PROJECT,
        entity=wandb_config.ENTITY,
        name="test-load.data",
        job_type="data-load")
    artifact = run.use_artifact(wandb_config.DATA_NAME + ":latest", type="dataset-suadd")
    artifact_dir = artifact.download(root=DATASET_PATH)

    dataset = ImageDataset(DATASET_PATH, size=None)
