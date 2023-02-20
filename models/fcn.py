# Fully Convolutional Network (FCN) for semantic segmentation
from torch import nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, FCN_ResNet50_Weights, FCN_ResNet101_Weights


class FCN(nn.Module):

    def __init__(self, num_classes, return_depth=False, backbone='50'):

        super().__init__()

        self.num_classes = num_classes
        self.return_depth = return_depth

        if backbone == 'resnet50':
            self.fcn = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, progress=True)
        elif backbone == 'resnet101':
            self.fcn = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT, progress=True)
        else:
            raise ValueError('backbone must be resnet50 or resnet101')

        # Only keep enough parameters for 1D inputs
        param = self.fcn.backbone.conv1.weight
        param = param[:, 0, :, :]
        param = param.unsqueeze(1)
        self.fcn.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fcn.backbone.conv1.weight = nn.Parameter(param)

        output_channels = num_classes
        if return_depth:
            output_channels += 1
        # Change the number of classes
        self.fcn.classifier[4] = nn.Conv2d(512, output_channels, kernel_size=1)
        self.fcn.aux_classifier[4] = nn.Conv2d(256, output_channels, kernel_size=1)

    def forward(self, x):
        output = self.fcn(x)['out']
        semantic = output[:, :self.num_classes, :, :]
        depth = None
        if self.return_depth:
            depth = output[:, self.num_classes:, :, :]
        return semantic, depth

if __name__ == "__main__":
    import torch
    model = FCN(num_classes=17, return_depth=True, backbone='50')
    x = torch.rand(1, 1, 256, 256)
    semantic, depth = model(x)
    print(semantic.shape, depth.shape)
    model = FCN(num_classes=17, return_depth=True, backbone='101')
    x = torch.rand(1, 1, 256, 256)
    semantic, depth = model(x)
    print(semantic.shape, depth.shape)