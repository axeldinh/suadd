import os
from tqdm import tqdm

import torch
from skimage.io import imsave
from torch.utils.data import DataLoader

from configs.globals import dataset_path, outputs_path
from transforms.transforms_1 import TransformSet1
from utils.datasets import ImageDataset
from utils.utils_image import make_overlay, unpatchify

if __name__ == "__main__":

    print("Sanity check for dataset loading")

    dataset = ImageDataset(dataset_path, transform=TransformSet1(patch_size=256))
    print("Dataset loaded successfully")

    train, val, test = dataset.split_dataset(0.8, 0.1)
    print("Dataset split successfully")

    # Check that dataloaders work
    try:
        train_loader = DataLoader(train, batch_size=2, num_workers=0, shuffle=True)
    except Exception as e:
        print("Error in making train dataloader:", e)

    try:
        val_loader = DataLoader(val, batch_size=2, num_workers=0, shuffle=False)
    except Exception as e:
        print("Error in making val dataloader:", e)

    try:
        test_loader = DataLoader(test, batch_size=2, num_workers=0, shuffle=False)
    except Exception as e:
        print("Error in making test dataloader:", e)
    print("Dataloaders created successfully")

    save_path = os.path.join(outputs_path, "check_dataset", "train")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving train images and overlays to {save_path}\n")

    for i, data in tqdm(enumerate(train)):

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

        if i > 10:
            break

    save_path = os.path.join(outputs_path, "check_dataset", "val")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving validation images and overlays to {save_path}\n")

    for i, data in tqdm(enumerate(val)):

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

        if i > 10:
            break

    save_path = os.path.join(outputs_path, "check_dataset", "test")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nSaving testing images and overlays to {save_path}\n")

    for i, data in tqdm(enumerate(test)):

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

        if i > 10:
            break
