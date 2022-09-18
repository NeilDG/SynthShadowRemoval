# -*- coding: utf-8 -*-
"""
Densenet as pre-trained block, with decoder block
Created on Mon Jun 29 14:30:24 2020

@author: delgallegon
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo

from utils import tensor_utils


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, nn.init.calculate_gain('tanh'))
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, in_features, norm_layer = nn.InstanceNorm2d):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
        # self.conv_block.register_forward_hook(self.on_conv_forward)

    def forward(self, x, densenet_layers):
        print("Res block shape: ", np.shape(x), " Feature output shape: ", np.shape(densenet_layers))
        densenet_layers = torch.reshape(densenet_layers, (densenet_layers.size()[0], 256, 31, 31))
        output = torch.cat([x, densenet_layers], 1)
        print("Resblock output cat: ", np.shape(output))

        return x + self.conv_block(x)

    def on_conv_forward(self, model, input, output):
        # torch.Size([64, 256, 31, 31])
        print("Output shape of res conv: ", np.shape(output))

def clamp(value, max):
    if(value > max):
        return max
    else:
        return value
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, upsampling_blocks=2, n_residual_blocks=6, has_dropout=True, norm_layer=nn.InstanceNorm2d):
        super(Generator, self).__init__()
        self.densenet_model = model_zoo.densenet161(True)
        self.densenet_model.eval()

        node_names = model_zoo.feature_extraction.get_graph_node_names(self.densenet_model)
        print("DENSENET Layers: ", node_names)
        self.feature_extractor = model_zoo.feature_extraction.create_feature_extractor(self.densenet_model, ["features.denseblock4.denselayer24.relu2"])

        encoding = []
        blocks = []
        decoding = []

        # Initial convolution block
        encoding = [nn.ReflectionPad2d(2),
                 nn.Conv2d(input_nc, 64, 8),
                 norm_layer(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(upsampling_blocks):
            encoding += [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                      norm_layer(out_features),
                      nn.ReLU(inplace=True)
                      ]

            if (has_dropout):
                encoding += [nn.Dropout2d(p=0.4)]
            in_features = out_features
            out_features = clamp(in_features * 2, 32768)

        # Residual blocks
        for _ in range(n_residual_blocks):
            blocks += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(upsampling_blocks):
            decoding += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=1),
                      norm_layer(out_features),
                      nn.ReLU(inplace=True)]

            if (has_dropout):
                decoding += [nn.Dropout2d(p=0.4)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        decoding += [nn.ReflectionPad2d(4),
                  nn.Conv2d(64, output_nc, 8),
                  nn.Tanh()]

        self.encoding = nn.Sequential(*encoding)
        # self.blocks = nn.Sequential(*blocks)
        self.blocks = blocks
        self.decoding = nn.Sequential(*decoding)

        self.encoding.apply(xavier_weights_init)
        # self.blocks.apply(xavier_weights_init)
        self.decoding.apply(xavier_weights_init)

    def forward(self, x):
        with torch.no_grad():
            output = self.feature_extractor(x)
            layer24 = output["features.denseblock4.denselayer24.relu2"]
            # layer24 = layer24.view(layer24.size()[0], 256, 31, 31)

        x = self.encoding(x)
        for block in self.blocks:
            x = block(x, layer24)

        y = self.decoding(x)

        return y
