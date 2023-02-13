from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, **args):
        super().__init__()

    def forward(self, semantic_out, depth_out, semantic, depth):
        raise NotImplementedError

    def set_weight(self, weight):
        raise NotImplementedError
