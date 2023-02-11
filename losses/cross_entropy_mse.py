import torch
from torch import nn


class CrossEntropyMSE(nn.Module):
    def __init__(self, coeff_depth=1.0):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.coeff_depth = coeff_depth

    def forward(self, semantic_out, depth_out, semantic, depth):
        if semantic.dim() == 4:
            semantic = semantic.squeeze(1)
        semantic_loss = self.semantic_loss(semantic_out, semantic.long())
        depth_loss = 0
        # Remove the nans from the depth
        if depth is not None:
            mask = ~torch.isnan(depth)
            depth = depth[mask]
            depth_out = depth_out[mask]
            depth_loss = self.depth_loss(depth_out, depth)
        output = {
            "semantic_loss": semantic_loss,
            "depth_loss": depth_loss,
            "loss": semantic_loss + depth_loss * self.coeff_depth
        }
        return output
