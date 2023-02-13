import torch
from torchvision.ops import sigmoid_focal_loss

from losses.base_loss import BaseLoss


class FocalLoss(BaseLoss):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_loss = sigmoid_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, semantic_out, depth_out, semantic, depth):
        num_classes = semantic_out.shape[1]
        height = semantic_out.shape[2]
        width = semantic_out.shape[3]

        # Make semantic with shape (B, H, W) to size (B, C, H, W) where C is the number of classes
        semantic_bin = torch.zeros((semantic.shape[0], num_classes, height, width), dtype=torch.float32)
        semantic_bin = semantic_bin.to(semantic.device)
        for c in semantic.unique():
            semantic_bin[:, c, :, :] = (semantic == c).float()

        focal_loss = self.focal_loss(semantic_out, semantic_bin, alpha=self.alpha, gamma=self.gamma)

        output = {
            "semantic_loss": focal_loss,
            "loss": focal_loss
        }

        return output
