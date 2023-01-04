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

DATASET_PREFIX_7_PATH = "E:/SynthWeather Dataset 7/"
DATASET_ALBEDO_7_PATH = "E:/SynthWeather Dataset 7/albedo/"

# PATCH_IMAGE_SIZE = (64, 64)
TEST_IMAGE_SIZE = (256, 256)

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

UNLIT_VERSION = "synth2unlit_v1.00"
EMBEDDING_VERSION = "embedding_v1.01"
FFA_TRANSFER_VERSION = "synthplaces2sunny_v1.01"
MAPPER_VERSION = "rgb2albedo_v1.00"
IID_VERSION = "maps2rgb_rgb2maps_v2.00"
RELIGHTING_VERSION = "relighter_v1.00"
SHADING_VERSION = "rgb2shading_v7.00"
SHADOWMAP_VERSION = "rgb2shadowmap_v1.00"
SHADOWMAP_RELIGHT_VERSION = "shadow2relight_v1.00"

# ITERATION = "1"

# UNLIT_TRANSFER_CHECKPATH ='checkpoint/' + UNLIT_VERSION + "_" + ITERATION + '.pt'
# STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + STYLE_TRANSFER_VERSION + "_" + ITERATION + '.pt'
# EMBEDDING_CHECKPATH = 'checkpoint/' + EMBEDDING_VERSION + "_" + ITERATION + '.pt'
# MAPPER_CHECKPATH = 'checkpoint/' + MAPPER_VERSION + "_" + ITERATION + '.pt'
# IID_CHECKPATH = 'checkpoint/' + IID_VERSION + "_" + ITERATION + '.pt'
# RELIGHTING_CHECKPATH = 'checkpoint/' + RELIGHTING_VERSION + "_" + ITERATION + '.pt'
# SHADING_CHECKPATH = 'checkpoint/' + SHADING_VERSION + "_" + ITERATION + '.pt'
# SHADOWMAP_CHECKPATH = 'checkpoint/' + SHADOWMAP_VERSION + "_" + ITERATION + '.pt'
# SHADOWMAP_RELIGHT_CHECKPATH = 'checkpoint/' + SHADOWMAP_RELIGHT_VERSION + "_" + ITERATION + '.pt'
#
# ALBEDO_MASK_VERSION = "rgb2mask_v1.00"
# ALBEDO_MASK_CHECKPATH = 'checkpoint/' + ALBEDO_MASK_VERSION + "_" + ITERATION + '.pt'

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
early_stop_threshold = 500
min_epochs = 50
num_epochs = 100
disable_progress_bar = False

#Running on local = 0, Running on COARE = 1, Running on CCS server = 2
server_config = 0
debug_run = 0
num_workers = 12

albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
rgb_dir_ws_styled = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
rgb_dir_ns_styled = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"

rgb_dir_ws = ""
rgb_dir_ns = ""

unlit_dir = "E:/SynthWeather Dataset 8/unlit/"

ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"
mask_istd = "E:/ISTD_Dataset/test/test_B/*.png"
# ws_istd = "E:/ISTD_Dataset/train/train_A/*.png"
# ns_istd = "E:/ISTD_Dataset/train/train_C/*.png"

ws_srd = "E:/SRD_Test/srd/shadow/*.jpg"
ns_srd = "E:/SRD_Test/srd/shadow_free/*.jpg"
# ws_srd = "E:/SRD_REMOVAL_RESULTS/rawA/*.png"
# ns_srd = "E:/SRD_REMOVAL_RESULTS/rawC/*.png"

imgx_dir = "E:/Places Dataset/*.jpg"
# imgy_dir = "E:/SynthWeather Dataset 8/train_rgb/*/*.png"
imgy_dir = "E:/SynthWeather Dataset 8/albedo/*.png"

shadow_removal_version = "VXX.XX"
shadow_matte_network_version = "VXX.XX"
style_transfer_version = "VXX.XX"