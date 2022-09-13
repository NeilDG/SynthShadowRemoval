import torch
from torch import nn

from model import usi3d_gan, vanilla_cycle_gan


class RelitGenerator(nn.Module):
    def __init__(self, input_nc, num_blocks):
        super(RelitGenerator, self).__init__()

        params = {'dim': 64,  # number of filters in the bottommost layer
                  'mlp_dim': 256,  # number of filters in MLP
                  'style_dim': 8,  # length of style code
                  'n_layer': 3,  # number of layers in feature merger/splitor
                  'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                  'n_downsample': 2,  # number of downsampling layers in content encoder
                  'n_res': num_blocks,  # number of residual blocks in content encoder/decoder
                  'pad_type': 'reflect'}

        # self.gamma_G = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params)
        # self.beta_G = usi3d_gan.AdaINGen(input_dim=input_nc, output_dim=3, params=params)
        self.gamma_G = vanilla_cycle_gan.Generator(3, 3, norm="instance")
        # self.beta_G = vanilla_cycle_gan.Generator(3, 3, norm="instance")

    def forward(self, x):
        # return torch.clip((x * self.gamma_G(x)) + self.beta_G(x), -2.0, 2.0)
        # return self.gamma_G(x) + self.beta_G(x)
        return torch.clip(self.gamma_G(x), -1.0, 1.0)