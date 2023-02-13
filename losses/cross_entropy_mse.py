import torch
from torch import nn

from losses.base_loss import BaseLoss


class CrossEntropyMSE(BaseLoss):
    def __init__(self, coeff_depth=1.0, weight=None):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss(weight=weight)
        self.depth_loss = nn.MSELoss()
        self.coeff_depth = coeff_depth

    def forward(self, semantic_out, depth_out, semantic, depth):
        if semantic.dim() == 4:
            semantic = semantic.squeeze(1)
        semantic_loss = self.semantic_loss(semantic_out, semantic.long())
        loss = semantic_loss
        depth_loss = None
        # Remove the nans from the depth
        if depth_out is not None:
            mask = ~torch.isnan(depth)
            depth = depth[mask]
            depth_out = depth_out[mask]
            depth_loss = self.depth_loss(depth_out, depth)
            loss = semantic_loss + depth_loss * self.coeff_depth
        output = {
            "semantic_loss": semantic_loss,
            "loss": loss
        }

        if depth_loss is not None:
            output["depth_loss"] = depth_loss

        return output

    def set_weight(self, weight):
        self.semantic_loss.weight = weight
