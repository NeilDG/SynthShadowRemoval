# -*- coding: utf-8 -*-
import os

DATASET_PLACES_PATH = "E:/Places Dataset/"

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