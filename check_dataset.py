import argparse
import os
import sys

import torch
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import trange

from configs.experiments import load_config
from configs.globals import DATASET_PATH, OUTPUTS_PATH
from utils.datasets import ImageDataset
from utils.utils_image import make_overlay


def main(exp_id, command):
    config = load_config(exp_id, command=command, create_folder=False)
    transform = config["transform"](**config["transform_args"])

    print("Sanity check for dataset loading, using config for experiment ", exp_id)

    dataset = ImageDataset(DATASET_PATH, transform=transform, store_images=config["store_images"])
    print("Dataset loaded successfully")

    print("Calling dataset.prepare_dataset()...")
    dataset.prepare_dataset(config["train_ratio"], config["val_ratio"])
    print("Dataset prepared successfully")

    train, val, test = dataset.get_data_splits()
    print("Dataset split successfully")

    # Check that dataloaders work
    try:
        train_loader = DataLoader(train, batch_size=2, num_workers=0, shuffle=True)
        for batch in train_loader:
            break
    except Exception as e:
        print("Error in making train dataloader:", e)

    try:
        val_loader = DataLoader(val, batch_size=1, num_workers=0, shuffle=False)
        for batch in val_loader:
            break
    except Exception as e:
        print("Error in making val dataloader:", e)

    try:
        test_loader = DataLoader(test, batch_size=1, num_workers=0, shuffle=False)
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
        if i >= len(val):
            break
        data = val[i]
        save_eval_data(data, i, save_path, transform)

    save_path = os.path.join(OUTPUTS_PATH, "check_dataset", "test")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving testing images and overlays to {save_path}\n")

    for i in trange(10):
        if i >= len(test):
            break
        data = test[i]
        save_eval_data(data, i, save_path, transform)


def save_eval_data(data, i, save_path, transform):
    image = data["image"]
    semantic = data["semantic"]
    depth = data["depth"]
    image_shape = data["image_shape"][0]
    image = image.reshape(image.shape[0], image.shape[-2], image.shape[-1])
    semantic = semantic.reshape(semantic.shape[0], semantic.shape[-2], semantic.shape[-1])
    depth = depth.reshape(depth.shape[0], depth.shape[-2], depth.shape[-1])
    image = transform.eval_untransform(image, image_shape)
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
    command = "python " + " ".join(sys.argv)

    main(**vars(args), command=command)
