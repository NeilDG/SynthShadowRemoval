# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""
import numbers

import kornia
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import cv2
from torch.autograd import Variable
import torch
from utils import pytorch_colors
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity
import torchvision.transforms as transforms
import global_config
from torchvision.transforms import functional as transform_functional

# for attaching hooks on pretrained models
class SaveFeatures(nn.Module):
    features = None;

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn);

    def hook_fn(self, module, input, output):
        self.features = output;

    def close(self):
        self.hook.remove();


class CombineFeatures(nn.Module):
    features = None;

    def __init(self, m, features):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = features

    def hook_fn(self, module, input, output):
        self.features = self.features + output

    def close(self):
        self.hook.remove()


def normalize_to_matplotimg(img_tensor, batch_idx, std, mean):
    img = img_tensor[batch_idx, :, :, :].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)  # for properly displaying image in matplotlib

    img = ((img * std) + mean)  # normalize back to 0-1 range

    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_to_matplotimg(img_tensor, batch_idx):
    img = img_tensor[batch_idx, :, :, :].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)  # for properly displaying image in matplotlib

    img = cv2.convertScaleAbs(img, alpha=(255.0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def convert_to_opencv(img_tensor):
    img = img_tensor
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)

    return img

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def normalize_to_01(input_tensor):
    return (input_tensor * 0.5) + 0.5

def normalize_for_gan(input_tensor):
    return transform_functional.normalize(input_tensor, [0.5,], [0.5,])

# loads an image compatible with opencv
def load_image(file_path):
    img = cv2.imread(file_path)
    if (img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Image ", file_path, " not found.")
    return img


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def interpret_one_hot(b_input, colorize = True):
    b_input = b_input.transpose(0, 1)

    if(colorize):
        mask_red = torch.stack([torch.full_like(b_input[0], 0.0),
                                torch.full_like(b_input[0], 0.0),
                                torch.full_like(b_input[0], 255.0)], 0)

        mask_green = torch.stack([torch.full_like(b_input[0], 0.0),
                                  torch.full_like(b_input[0], 255.0),
                                  torch.full_like(b_input[0], 0.0)], 0)

        mask_blue = torch.stack([torch.full_like(b_input[0], 255.0),
                                 torch.full_like(b_input[0], 0.0),
                                 torch.full_like(b_input[0], 0.0)], 0)

        b_1 = 1 - b_input[0] * mask_red
        b_2 = b_input[1] * mask_green
        b_3 = b_input[2] * mask_blue
        b_4 = b_input[3] * (mask_red + mask_green)
        b_5 = b_input[4] * (mask_red + mask_blue)

        result = b_1 + b_2 + b_3 + b_4
        result = result.transpose(0, 1)
        return result

    else:
        b_1 = 1 - b_input[0]
        b_2 = b_input[1]
        b_3 = b_input[2] * (29.0 / 255.0)
        b_4 = b_input[3] * (51.0 / 255.0) - 0.5 #make it darker

        result = b_1 + b_2 + b_3 + b_4
        result = torch.unsqueeze(result, 1)

        return result

def merge_yuv_results_to_rgb(y_tensor, uv_tensor):
    uv_tensor = uv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)

    (u, v) = torch.chunk(uv_tensor, 2)
    yuv_tensor = torch.cat((y_tensor, u, v))
    rgb_tensor = pytorch_colors.lab_to_rgb(yuv_tensor.transpose(0, 1))
    # rgb_tensor = ((rgb_tensor * 0.5) + 0.5) #normalize back to 0-1 range
    rgb_tensor = ((rgb_tensor * 1.0) + 1.0)
    return rgb_tensor


def yuv_to_rgb(yuv_tensor):
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor)
    return rgb_tensor


def rgb_to_yuv(rgb_tensor):
    return pytorch_colors.rgb_to_yuv(rgb_tensor)


def change_yuv(y_tensor, yuv_tensor):
    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_tensor = y_tensor.transpose(0, 1)
    (y, u, v) = torch.chunk(yuv_tensor, 3)
    yuv_tensor = torch.cat((y_tensor, u, v))
    return yuv_tensor.transpose(0, 1)


def replace_dark_channel(rgb_tensor, dark_channel_old, dark_channel_new, alpha=0.7, beta=0.7):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)

    yuv_tensor = yuv_tensor.transpose(0, 1)
    dark_channel_old = dark_channel_old.transpose(0, 1)
    dark_channel_new = dark_channel_new.transpose(0, 1)

    (y, u, v) = torch.chunk(yuv_tensor, 3)

    # deduct old dark channel from all channels and add new one
    # r = r - dark_channel_old + dark_channel_new
    # g = g - dark_channel_old + dark_channel_new
    # b = b - dark_channel_old + dark_channel_new
    y = y - (dark_channel_old * alpha) + (dark_channel_new * beta)

    yuv_tensor = torch.cat((y, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor


def replace_y_channel(rgb_tensor, y_new):
    yuv_tensor = pytorch_colors.rgb_to_yuv(rgb_tensor)

    yuv_tensor = yuv_tensor.transpose(0, 1)
    y_new = y_new.transpose(0, 1)

    (y, u, v) = torch.chunk(yuv_tensor, 3)

    yuv_tensor = torch.cat((y_new, u, v))
    rgb_tensor = pytorch_colors.yuv_to_rgb(yuv_tensor.transpose(0, 1))
    return rgb_tensor

def get_y_channel(I):
    y, u, v = cv2.split(I)
    return y


def get_uv_channel(I):
    y, u, v = cv2.split(I)
    return cv2.merge((u, v))

def make_rgb(batch):
    batch = batch.transpose(0, 1)
    (b, g, r) = torch.chunk(batch, 3)
    batch = torch.cat((r, g, b))
    batch = batch.transpose(0, 1)
    return batch

def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - Variable(mean).cuda()


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + Variable(mean).cuda()


def imagenet_clamp_batch(batch, low, high):
    batch[:, 0, :, :].data.clamp_(low - 103.939, high - 103.939)
    batch[:, 1, :, :].data.clamp_(low - 116.779, high - 116.779)
    batch[:, 2, :, :].data.clamp_(low - 123.680, high - 123.680)


# computes a z_signal based on image size. Image size must always be a power of 2 and greater than 16x16.
def compute_z_signal(value, batch_size, image_size):
    z_size = (int(image_size[0] / 16), int(image_size[1] / 16))
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, z_size[0], z_size[1]))
    return z_signal


# computes a z signal to be conacated with another image tensor.
def compute_z_signal_concat(value, batch_size, image_size):
    torch.manual_seed(value)
    z_signal = torch.randn((batch_size, 100, image_size[0], image_size[1]))
    return z_signal

#
# def measure_ssim(img1, img2):
#     # preprocessing
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
#
#     return structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, sigma=1.5)

# def produce_albedo(rgb_tensor, shading_tensor):
#     rgb_tensor = rgb_tensor.transpose(0, 1)
#     shading_tensor = shading_tensor.transpose(0, 1)
#
#     rgb_tensor = (rgb_tensor * 0.5) + 0.5
#     shading_tensor = (shading_tensor * 0.5) + 0.5
#
#     #clip values to avoid overflow
#     shading_tensor = torch.clip(shading_tensor, 0.00001, 1.0)
#
#     #derive albedo manually
#     albedo_tensor = torch.full_like(rgb_tensor, 0, requires_grad=False)
#     albedo_tensor[0] = rgb_tensor[0] / shading_tensor
#     albedo_tensor[1] = rgb_tensor[1] / shading_tensor
#     albedo_tensor[2] = rgb_tensor[2] / shading_tensor
#     albedo_tensor = torch.clip(albedo_tensor, 0.00001, 1.0)
#
#     albedo_tensor = albedo_tensor.transpose(0, 1)
#     return albedo_tensor
#
# def produce_rgb(albedo_tensor, shading_tensor, light_color):
#     albedo_tensor = albedo_tensor.transpose(0, 1)
#     shading_tensor = shading_tensor.transpose(0, 1)
#     light_color = torch.from_numpy(np.asarray(light_color.split(","), dtype=np.int32))
#
#     # normalize/remove normalization
#     albedo_tensor = (albedo_tensor * 0.5) + 0.5
#     shading_tensor = (shading_tensor * 0.5) + 0.5
#
#     light_color = light_color / 255.0
#
#     # clip values to avoid overflow
#     shading_tensor = torch.clip(shading_tensor, 0.00001, 1.0)
#
#     rgb_img_like = torch.full_like(albedo_tensor, 0)
#     rgb_img_like[0] = torch.clip(albedo_tensor[0] * shading_tensor * light_color[0], 0.0, 1.0)
#     rgb_img_like[1] = torch.clip(albedo_tensor[1] * shading_tensor * light_color[1], 0.0, 1.0)
#     rgb_img_like[2] = torch.clip(albedo_tensor[2] * shading_tensor * light_color[2], 0.0, 1.0)
#
#     rgb_img_like = rgb_img_like.transpose(0, 1)
#     return rgb_img_like


def load_metric_compatible_albedo(img_path, cvt_color: int, normalize: bool, convert_to_tensor: bool, size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cvt_color)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if (normalize):
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img_mask = cv2.inRange(img[:, :, 0], 0.0, 0.01)  # mask for getting zero pixels to be excluded
    img_mask = np.asarray([img_mask, img_mask, img_mask])
    img_mask = np.moveaxis(img_mask, 0, 2)

    img_ones = np.full_like(img_mask, 1.0)
    img = np.clip(img + (img_ones * img_mask), 0.01, 1.0)

    if (convert_to_tensor):
        transform_op = transforms.ToTensor()
        tensor_img = transform_op(img)
        tensor_img = torch.unsqueeze(tensor_img, 0)
        return tensor_img

    else:
        return img

def load_metric_compatible_img(img_path, cvt_color:int, normalize:bool, convert_to_tensor:bool, size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cvt_color)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if(normalize):
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if(convert_to_tensor):
        transform_op = transforms.ToTensor()
        tensor_img = transform_op(img)
        tensor_img = torch.unsqueeze(tensor_img, 0)
        return tensor_img

    else:
        return img

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
