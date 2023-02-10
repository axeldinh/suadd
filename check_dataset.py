import os

import torch
from skimage.io import imsave

from configs.paths import dataset_path, outputs_path
from transforms.transforms_1 import TransformSet1
from utils.datasets import ImageDataset
from utils.utils_image import make_overlay, unpatchify

if __name__ == "__main__":

    dataset = ImageDataset(dataset_path, transform=TransformSet1(patch_size=256))
    train, val, test = dataset.split_dataset(0.8, 0.1)

    save_path = os.path.join(outputs_path, "check_dataset", "train")
    os.makedirs(save_path, exist_ok=True)

    for i, batch in enumerate(train):

        image = batch["image"]
        semantic = batch["semantic"]
        depth = batch["depth"]

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

    for i, batch in enumerate(val):

        image = batch["image"]
        semantic = batch["semantic"]
        depth = batch["depth"]
        image_shape = batch["image_shape"]

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

    for i, batch in enumerate(test):

        image = batch["image"]
        semantic = batch["semantic"]
        depth = batch["depth"]
        image_shape = batch["image_shape"]

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
