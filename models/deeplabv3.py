from torch import nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large


class DeepLabV3(nn.Module):

    def __init__(self, num_classes, return_depth=False, backbone="resnet50"):

        super().__init__()

        self.num_classes = num_classes
        self.return_depth = return_depth
        self.backbone = backbone

        if backbone == "resnet50":
            self.deeplab = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, progress=True)
        elif backbone == "resnet101":
            self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=True)
        elif backbone == "mobilenet_v3_large":
            self.deeplab = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
                                                        progress=True)
        else:
            raise ValueError("backbone must be resnet50, resnet101 or mobilenet_v3_large")

        output_channels = num_classes
        if return_depth:
            output_channels += 1
        if "resnet" in backbone:
            # Only keep enough parameters for 1D inputs
            param = self.deeplab.backbone.conv1.weight
            param = param[:, 0, :, :]
            param = param.unsqueeze(1)
            self.deeplab.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.deeplab.backbone.conv1.weight = nn.Parameter(param)
            # Change the number of classes
            self.deeplab.classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))
            self.deeplab.aux_classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))
        elif "mobilenet" in backbone:
            param = self.deeplab.backbone['0'][0].weight
            param = param[:, 0, :, :]
            param = param.unsqueeze(1)
            self.deeplab.backbone['0'][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.deeplab.backbone['0'][0].weight = nn.Parameter(param)
            self.deeplab.classifier[4] = nn.Conv2d(256, output_channels, kernel_size=(1, 1), stride=(1, 1))
            self.deeplab.aux_classifier[4] = nn.Conv2d(10, output_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        output = self.deeplab(x)['out']
        semantic = output[:, :self.num_classes, :, :]
        depth = None
        if self.return_depth:
            depth = output[:, self.num_classes:, :, :]
        return semantic, depth
