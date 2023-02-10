import torch
from torchvision.utils import draw_segmentation_masks


def make_overlay(image, annotation):
    """
    Make an overlay of the image and the annotation
    :param image: torch gray images (1, height, width)
    :param annotation: torch gray images (1, height, width)
    :return: overlay of the image and the annotation (3, height, width)
    """
    palette = [
        (148, 218, 255),  # light blue, WATER
        (85, 85, 85),  # almost black, ASPHALT
        (200, 219, 190),  # light green, GRASS
        (166, 133, 226),  # purple, HUMAN
        (255, 171, 225),  # pink, ANIMAL
        (40, 150, 114),  # green, HIGH VEGETATION
        (234, 144, 133),  # orange, GROUND VEHICLE
        (89, 82, 96),  # dark gray, FACADE
        (255, 255, 0),  # yellow, WIRE
        (110, 87, 121),  # dark purple, GARDEN FURNITURE
        (205, 201, 195),  # light gray, CONCRETE
        (212, 80, 121),  # medium red, ROOF
        (159, 135, 114),  # light brown, GRAVEL
        (102, 90, 72),  # dark brown, SOIL
        (255, 255, 102),  # bright yellow, PRIMEAIR PATTERN
        (251, 247, 240),  # almost white, SNOW
        (0, 0, 0),  # black, UNKNOWN
    ]

    classes = torch.unique(annotation)
    # Sort the classes so that the overlay is always the same
    classes = torch.sort(classes)[0]
    palette = [palette[int(cls)] for cls in classes]
    masks = torch.zeros_like(annotation, dtype=bool)
    masks = masks.repeat(classes.shape[0], 1, 1)
    for j, cls in enumerate(classes):
        masks[j][annotation[0] == cls] = True
    overlay = draw_segmentation_masks(image.repeat(3, 1, 1), masks, alpha=0.5, colors=palette)
    return overlay


def patchify(image, patch_size):
    """
    Make patches out of an image of a desired size. Some patches may overlap.
    :param image: torch gray images (1, height, width)
    :param patch_size: size of the patches
    :return: patches of the image (n_patches, patch_size, patch_size)
    """
    image_height = image.shape[1]
    image_width = image.shape[2]

    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    if image_height % patch_size != 0:
        num_patches_height += 1
    if image_width % patch_size != 0:
        num_patches_width += 1

    patches = torch.zeros((num_patches_height * num_patches_width, patch_size, patch_size))
    for i in range(num_patches_height * num_patches_width):
        x = i % num_patches_width
        y = i // num_patches_width
        if (y + 1) * patch_size > image_height:
            slice_y = slice(image_height - patch_size, image_height)
        else:
            slice_y = slice(y * patch_size, (y + 1) * patch_size)
        if (x + 1) * patch_size > image_width:
            slice_x = slice(image_width - patch_size, image_width)
        else:
            slice_x = slice(x * patch_size, (x + 1) * patch_size)

        patches[i] = image[:, slice_y, slice_x]

    return patches


def unpatchify(patches, image_shape):
    """
    Reconstruct an image from patches. The values are averaged if the same pixel is covered by multiple patches.
    :param patches: patches of the image (n_patches, patch_size, patch_size)
    :param image_size: size of the image
    :return: torch gray images (1, height, width)
    """

    image_height = image_shape[1]
    image_width = image_shape[2]
    patch_size = patches.shape[1]

    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    if image_height % patch_size != 0:
        num_patches_height += 1
    if image_width % patch_size != 0:
        num_patches_width += 1

    image = torch.zeros(image_shape)
    mask = torch.zeros(image_shape)
    for i in range(num_patches_height * num_patches_width):
        x = i % num_patches_width
        y = i // num_patches_width
        if (y + 1) * patch_size > image_height:
            slice_y = slice(image_height - patch_size, image_height)
        else:
            slice_y = slice(y * patch_size, (y + 1) * patch_size)
        if (x + 1) * patch_size > image_width:
            slice_x = slice(image_width - patch_size, image_width)
        else:
            slice_x = slice(x * patch_size, (x + 1) * patch_size)
        image[:, slice_y, slice_x] += patches[i]
        mask[:, slice_y, slice_x] += 1

    return image / mask
