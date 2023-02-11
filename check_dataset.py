import argparse
import os

import torch
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import trange

from configs.experiments import load_config
from configs.globals import DATASET_PATH, OUTPUTS_PATH
from utils.datasets import ImageDataset
from utils.utils_image import make_overlay, unpatchify


def main(exp_id, command):
    config = load_config(exp_id, command=command)
    transform = config["transform"](**config["transform_args"])

    print("Sanity check for dataset loading, using config for experiment ", exp_id)

    dataset = ImageDataset(DATASET_PATH, transform=transform)
    print("Dataset loaded successfully")

    train, val, test = dataset.split_dataset(0.8, 0.1)
    print("Dataset split successfully")

    # Check that dataloaders work
    try:
        train_loader = DataLoader(train, batch_size=2, num_workers=0, shuffle=True)
        for batch in train_loader:
            break
    except Exception as e:
        print("Error in making train dataloader:", e)

    try:
        val_loader = DataLoader(val, batch_size=2, num_workers=0, shuffle=False)
        for batch in val_loader:
            break
    except Exception as e:
        print("Error in making val dataloader:", e)

    try:
        test_loader = DataLoader(test, batch_size=2, num_workers=0, shuffle=False)
        for batch in test_loader:
            break
    except Exception as e:
        print("Error in making test dataloader:", e)
    print("Dataloaders created successfully")

    save_path = os.path.join(OUTPUTS_PATH, "check_dataset", "train")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving train images and overlays to {save_path}\n")

    for i in trange(10):
        data = train[i]

        image = data["image"]
        semantic = data["semantic"]
        depth = data["depth"]

        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).type(torch.uint8)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).type(torch.uint8)

        overlay = make_overlay(image, semantic)

        image = image.numpy().transpose(1, 2, 0)
        overlay = overlay.numpy().transpose(1, 2, 0)
        depth = depth.numpy().transpose(1, 2, 0)

        imsave(os.path.join(save_path, f"{i}_image.png"), image)
        imsave(os.path.join(save_path, f"{i}_overlay.png"), overlay)
        imsave(os.path.join(save_path, f"{i}_depth.png"), depth)

    save_path = os.path.join(OUTPUTS_PATH, "check_dataset", "val")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving validation images and overlays to {save_path}\n")

    for i in trange(10):
        data = val[i]

        image = data["image"]
        semantic = data["semantic"]
        depth = data["depth"]
        image_shape = data["image_shape"]

        image = unpatchify(image, image_shape)
        semantic = unpatchify(semantic, image_shape)
        depth = unpatchify(depth, image_shape)

        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).type(torch.uint8)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).type(torch.uint8)

        overlay = make_overlay(image, semantic)
        image = image.numpy().transpose(1, 2, 0)
        overlay = overlay.numpy().transpose(1, 2, 0)
        depth = depth.numpy().transpose(1, 2, 0)

        imsave(os.path.join(save_path, f"{i}_image.png"), image)
        imsave(os.path.join(save_path, f"{i}_overlay.png"), overlay)
        imsave(os.path.join(save_path, f"{i}_depth.png"), depth)

    save_path = os.path.join(OUTPUTS_PATH, "check_dataset", "test")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving testing images and overlays to {save_path}\n")

    for i in trange(10):
        data = test[i]

        image = data["image"]
        semantic = data["semantic"]
        depth = data["depth"]
        image_shape = data["image_shape"]

        image = unpatchify(image, image_shape)
        semantic = unpatchify(semantic, image_shape)
        depth = unpatchify(depth, image_shape)

        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).type(torch.uint8)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).type(torch.uint8)

        overlay = make_overlay(image, semantic)
        image = image.numpy().transpose(1, 2, 0)
        overlay = overlay.numpy().transpose(1, 2, 0)
        depth = depth.numpy().transpose(1, 2, 0)

        imsave(os.path.join(save_path, f"{i}_image.png"), image)
        imsave(os.path.join(save_path, f"{i}_overlay.png"), overlay)
        imsave(os.path.join(save_path, f"{i}_depth.png"), depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", "-e", type=int, required=True, help="Experiment ID")
    args = parser.parse_args()

    # Recover the complete command entered by the user
    command = " ".join(["python"] + [f"--{k} {v}" for k, v in vars(args).items() if v is not None])

    main(**vars(args), command=command)
