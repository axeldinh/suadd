
from torchvision.utils import draw_segmentation_masks
import torch

def make_overlay(image, annotation):
    classes = torch.unique(annotation)
    masks = torch.zeros_like(annotation, dtype=bool)
    masks = masks.repeat(classes.shape[0], 1, 1)
    for j, cls in enumerate(classes):
        masks[j][annotation[0] == cls] = True
    overlay = draw_segmentation_masks(image.repeat(3, 1, 1), masks, alpha=0.5)
    return overlay