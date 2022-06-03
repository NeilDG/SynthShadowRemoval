# -*- coding: utf-8 -*-
"""
An Embedding Encoder-Decoder inspired from Vanilla CycleGAN
Created on Mon Jun 29 14:30:24 2020

@author: delgallegon
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def normal_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

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

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, downsampling_blocks = 2, n_residual_blocks=6, has_dropout = True):
        super(EmbeddingNetwork, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(2),
                    nn.Conv2d(input_nc, 64, 8),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(downsampling_blocks):
            model += [  nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)
                    ]

            if(has_dropout):
                model +=[nn.Dropout2d(p = 0.1)]
            in_features = out_features
            out_features = clamp(in_features*2, 8192)

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        self.encoder = nn.Sequential(*model)
        self.encoder.apply(weights_init)

        model = []
        # Upsampling
        out_features = in_features // 2
        for _ in range(downsampling_blocks):
            model += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]

            if (has_dropout):
                model += [nn.Dropout2d(p=0.1)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(4),
                  nn.Conv2d(64, output_nc, 8),
                  nn.Tanh()]

        self.decoder = nn.Sequential(*model)
        self.decoder.apply(weights_init)


    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)

    def get_decoding(self, x):
        return self.decoder(x)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class EmbeddingNetworkFFA(nn.Module):
    def __init__(self, blocks):
        super(EmbeddingNetworkFFA, self).__init__()
        self.gps = 3
        self.dim = 64
        conv = default_conv
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def get_embedding(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]

        return w, res1, res2, res3

    def get_decoding(self, x1, w, res1, res2, res3):
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)

        return x + x1

    def forward(self, x1):
        w, res1, res2, res3 = self.get_embedding(x1)
        return self.get_decoding(x1, w, res1, res2, res3)

class DecodingNetworkFFA(nn.Module):
    def __init__(self):
        super(DecodingNetworkFFA, self).__init__()
        self.dim = 64
        self.palayer = PALayer(self.dim)
        conv = default_conv
        kernel_size = 3
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.post = nn.Sequential(*post_precess)
        self.post.apply(normal_weights_init)

    def get_decoding(self, x1, w, res1, res2, res3):
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)

        return x + x1

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
