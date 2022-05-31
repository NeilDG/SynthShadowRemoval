# -*- coding: utf-8 -*-
import os

DATASET_PLACES_PATH = "E:/Places Dataset/"
DATASET_VEMON_PATH = "E:/VEMON Dataset/frames/"
DATASET_DIV2K_PATH_PATCH = "E:/Div2k - Patch/"
DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"
DATASET_WEATHER_SUNNY_PATH = "E:/SynthWeather Dataset/sunny/"
DATASET_WEATHER_NIGHT_PATH = "E:/SynthWeather Dataset/night/"
DATASET_WEATHER_CLOUDY_PATH = "E:/SynthWeather Dataset/cloudy/"
DATASET_WEATHER_STYLED_PATH = "E:/SynthWeather Dataset/styled/"
DATASET_WEATHER_DEFAULT_PATH = "E:/SynthWeather Dataset/default/"
DATASET_WEATHER_SEGMENT_PATH = "E:/SynthWeather Dataset/segmentation/"

DATASET_RGB_PATH = "E:/SynthWeather Dataset 2/default/"
DATASET_ALBEDO_PATH = "E:/SynthWeather Dataset 2/albedo/"
DATASET_NORMAL_PATH = "E:/SynthWeather Dataset 2/normal/"
DATASET_SPECULAR_PATH = "E:/SynthWeather Dataset 2/specular/"
DATASET_SMOOTHNESS_PATH = "E:/SynthWeather Dataset 2/smoothness/"
DATASET_LIGHTMAP_PATH = "E:/SynthWeather Dataset 2/lightmap/"

DATASET_RGB_DECOMPOSE_PATH = "E:/SynthWeather Dataset 3/rgb/"
DATASET_SHADING_DECOMPOSE_PATH = "E:/SynthWeather Dataset 3/shading/"
DATASET_ALBEDO_DECOMPOSE_PATH = "E:/SynthWeather Dataset 3/albedo/"

DATASET_PREFIX_6_PATH = "E:/SynthWeather Dataset 6/"
DATASET_ALBEDO_6_PATH = "E:/SynthWeather Dataset 6/albedo/"

# PATCH_IMAGE_SIZE = (64, 64)
TEST_IMAGE_SIZE = (256, 256)

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

STYLE_TRANSFER_VERSION = "synth2rgb_v1.00"
EMBEDDING_VERSION = "embedding_v1.00"
FFA_TRANSFER_VERSION = "synthplaces2sunny_v1.01"
MAPPER_VERSION = "rgb2albedo_v1.00"
IID_VERSION = "maps2rgb_rgb2maps_v2.00"
RELIGHTING_VERSION = "relighter_v1.00"
SHADING_VERSION = "rgb2shading_v7.00"
SHADOWMAP_VERSION = "rgb2shadowmap_v1.00"
SHADOWMAP_RELIGHT_VERSION = "shadow2relight_v1.00"

ITERATION = "1"

STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + STYLE_TRANSFER_VERSION + "_" + ITERATION + '.pt'
EMBEDDING_CHECKPATH = 'checkpoint/' + EMBEDDING_VERSION + "_" + ITERATION + '.pt'
MAPPER_CHECKPATH = 'checkpoint/' + MAPPER_VERSION + "_" + ITERATION + '.pt'
IID_CHECKPATH = 'checkpoint/' + IID_VERSION + "_" + ITERATION + '.pt'
RELIGHTING_CHECKPATH = 'checkpoint/' + RELIGHTING_VERSION + "_" + ITERATION + '.pt'
SHADING_CHECKPATH = 'checkpoint/' + SHADING_VERSION + "_" + ITERATION + '.pt'
SHADOWMAP_CHECKPATH = 'checkpoint/' + SHADOWMAP_VERSION + "_" + ITERATION + '.pt'
SHADOWMAP_RELIGHT_CHECKPATH = 'checkpoint/' + SHADOWMAP_RELIGHT_VERSION + "_" + ITERATION + '.pt'

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
G_ADV_LOSS_KEY = "g_adv"
LIKENESS_LOSS_KEY = "likeness"
LPIP_LOSS_KEY = "lpip"
SSIM_LOSS_KEY = "ssim"
PSNR_LOSS_KEY = "colorshift"
SMOOTHNESS_LOSS_KEY = "smoothness"
EDGE_LOSS_KEY = "edge"
RECONSTRUCTION_LOSS_KEY = "reconstruction"

D_OVERALL_LOSS_KEY = "d_loss"
D_A_REAL_LOSS_KEY = "d_real_a"
D_A_FAKE_LOSS_KEY = "d_fake_a"
D_B_REAL_LOSS_KEY = "d_real_b"
D_B_FAKE_LOSS_KEY = "d_fake_b"

LAST_METRIC_KEY = "last_metric"

plot_enabled = 1
early_stop_threshold = 1000
min_epochs = 50
num_epochs = 500

#Running on local = 0, Running on COARE = 1, Running on CCS server = 2
server_config = 0
num_workers = 12

imgy_dir = "E:/SynthWeather Dataset 6/azimuth/*/rgb/*.png"
imgx_dir = "E:/GTAV_Processed/images/*.png"
imgx_dir_test = "E:/SynthWeather Dataset 6/azimuth/*/rgb/*.png"
imgy_dir_test = "E:/GTAV_Processed/images/*.png"

# imgx_dir = "E:/Image Transfer - Patches/trainA/*.png"
# imgy_dir = "E:/Image Transfer - Patches/trainB/*.png"
# imgx_dir_test = "E:/Image Transfer - Patches/testA/*.png"
# imgy_dir_test = "E:/Image Transfer - Patches/testB/*.png"
