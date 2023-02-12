# Implementation of a U-Net model

import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, conv_size=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_size = conv_size
        self.padding = (conv_size - 1) // 2

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=conv_size, padding=self.padding)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=conv_size, padding=self.padding)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return x


class DecodingLayer(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=int(in_channels / 2),
                                                 kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=int(in_channels / 2))

    def forward(self, x, skip):
        x = torch.concatenate([self.transpose_conv(x), skip], dim=1)
        x = self.double_conv(x)

        return x


class UNet(nn.Module):

    def __init__(self, channels_in, first_layer_channels, depth, num_classes, conv_size=3, return_depth=False):
        super().__init__()

        assert isinstance(depth, int) and depth > 0, f"Depth={depth} must be a positive integer"
        assert isinstance(channels_in, int) and channels_in > 0, f"channels_in={channels_in} must be a positive integer"
        assert isinstance(first_layer_channels,
                          int) and first_layer_channels > 0, f"first_layer_channels={first_layer_channels}" \
                                                             f" must be a positive integer"
        assert isinstance(num_classes, int) and num_classes > 0, f"number_classes={num_classes} should be " \
                                                                 f"a positive integer"

        self.channels_in = channels_in
        self.first_layer_channels = first_layer_channels
        self.encoding_layers = nn.ModuleList([DoubleConv(self.channels_in, self.first_layer_channels)])
        self.depth = depth
        self.num_classes = num_classes
        self.out_channels = num_classes
        self.conv_size = conv_size
        if return_depth:
            self.out_channels += 1
        self.return_depth = return_depth

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.encoding_layers += nn.ModuleList([
            DoubleConv(self.first_layer_channels * (2 ** i), self.first_layer_channels * (2 ** (i + 1)))
            for i in range(depth)
        ])

        self.decoding_layers = nn.ModuleList([
            DecodingLayer(self.first_layer_channels * (2 ** i))
            for i in range(depth, 0, -1)
        ])

        self.last_layer = nn.Conv2d(in_channels=first_layer_channels, out_channels=self.out_channels, kernel_size=3,
                                    padding=1)

    def forward(self, x):

        skips = []

        # Encoding
        for i in range(len(self.encoding_layers)):
            x = self.encoding_layers[i](x)
            if i < self.depth:
                skips.append(x.clone())
                x = self.max_pool(x)

        skips = skips[::-1]

        # Decoding
        for i in range(len(self.decoding_layers)):
            x = self.decoding_layers[i](x, skips[i])

        x = self.last_layer(x)

        semantic = x[:, :self.num_classes, :, :]
        depth = None
        if self.return_depth:
            depth = x[:, -1, :, :].unsqueeze(1)

        return semantic, depth
