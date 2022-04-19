# -*- coding: utf-8 -*-
"""
Modified implementation of the following model:
Gonzalez-Garcia, A., Van De Weijer, J., & Bengio, Y. (2018).
Image-to-image translation for cross-domain disentanglement. Advances in neural information processing systems, 31.

@author: delgallegon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def clamp(value, max):
    if (value > max):
        return max
    else:
        return value


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GradReverseLayer(nn.Module):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return (-grad_output)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, downsampling_blocks=2, n_residual_blocks=6, has_dropout=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(2),
                 nn.Conv2d(input_nc, 64, 8),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(downsampling_blocks):
            model += [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU(inplace=True)
                      ]

            if (has_dropout):
                model += [nn.Dropout2d(p=0.4)]
            in_features = out_features
            out_features = clamp(in_features * 2, 8192)

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        #add GRL
        model += [GradReverseLayer()]

        self.encoding_block = nn.Sequential(*model)

        model = []
        # Upsampling
        out_features = in_features // 2
        for _ in range(downsampling_blocks):
            model += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU(inplace=True)]

            if (has_dropout):
                model += [nn.Dropout2d(p=0.4)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(4),
                  nn.Conv2d(64, output_nc, 8),
                  nn.Tanh()]

        self.decoding_block = nn.Sequential(*model)

    def forward(self, x):
        return self.decoding_block(self.encoding_block(x))

    def get_encoding(self, x):
        return self.encoding_block(x)

    def get_decoding(self, feature_x):
        return self.decoding_block(feature_x)