# -*- coding: utf-8 -*-
import os

DATASET_PLACES_PATH = "E:/Places Dataset/"
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
disable_progress_bar = False

server_config = -1
num_workers = -1

albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
rgb_dir_ws_styled = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
rgb_dir_ns_styled = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"

rgb_dir_ws = ""
rgb_dir_ns = ""

unlit_dir = "E:/SynthWeather Dataset 8/unlit/"

ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"
# ws_istd = "E:/ISTD_Dataset/train/train_A/*.png"
# ns_istd = "E:/ISTD_Dataset/train/train_C/*.png"

ws_srd = "X:/SRD_Test/srd/shadow/*.jpg"
ns_srd = "X:/SRD_Test/srd/shadow_free/*.jpg"
mask_srd = "X:/SRD_Test/srd/mask/*.jpg"
# ws_srd = "E:/SRD_REMOVAL_RESULTS/rawA/*.png"
# ns_srd = "E:/SRD_REMOVAL_RESULTS/rawC/*.png"

#NOTE that USR is unpaired
ws_usr = "E:/USR Shadow Dataset/shadow_train/*.jpg"
ns_usr = "E:/USR Shadow Dataset/shadow_free/*.jpg"
usr_test = "E:/USR Shadow Dataset/shadow_test/*.jpg"

imgx_dir = "E:/Places Dataset/*.jpg"
# imgy_dir = "E:/SynthWeather Dataset 8/train_rgb/*/*.png"
imgy_dir = "E:/SynthWeather Dataset 8/albedo/*.png"

a_path = ""
b_path = ""

# shadow_removal_version = "VXX.XX"
# shadow_matte_network_version = "VXX.XX"
style_transfer_version = "VXX.XX"
st_iteration = -1

sm_network_version = "VXX.XX"
sm_iteration = -1
ns_network_version = "VXX.XX"
ns_iteration = -1

loaded_network_config = None
sm_network_config = None
ns_network_config = None

img_to_load = -1
load_size = -1
batch_size = -1
test_size = -1
train_mode = "all"
last_epoch_sm = 0
last_epoch_ns = 0
last_epoch_st = 0
last_iteration_ns = 0
dataset_target = ""
cuda_device = ""
save_images = 0
save_every_epoch = 5
epoch_to_load = 0
load_per_epoch = False
load_per_sample = False
load_best = False


