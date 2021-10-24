# -*- coding: utf-8 -*-
import os

DATASET_PLACES_PATH = "E:/Places Dataset/"
DATASET_DIV2K_PATH_PATCH = "E:/Div2k - Patch/"
DATASET_DIV2K_PATH = "E:/DIV2K_train_HR/"
DATASET_WEATHER_SUNNY_PATH = "E:/SynthWeather Dataset/sunny/"
DATASET_WEATHER_NIGHT_PATH = "E:/SynthWeather Dataset/night/"
DATASET_WEATHER_CLOUDY_PATH = "E:/SynthWeather Dataset/cloudy/"

PATCH_IMAGE_SIZE = (32, 32)
TEST_IMAGE_SIZE = (256, 256)

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

STYLE_TRANSFER_VERSION = "places2sunnyweather_v1.00"

ITERATION = "1"
STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + STYLE_TRANSFER_VERSION + "_" + ITERATION + '.pt'

# dictionary keys
G_LOSS_KEY = "g_loss"
IDENTITY_LOSS_KEY = "id"
CYCLE_LOSS_KEY = "cyc"
G_ADV_LOSS_KEY = "g_adv"
LIKENESS_LOSS_KEY = "likeness"
REALNESS_LOSS_KEY = "realness"
PSNR_LOSS_KEY = "colorshift"
SMOOTHNESS_LOSS_KEY = "smoothness"
EDGE_LOSS_KEY = "edge"

D_OVERALL_LOSS_KEY = "d_loss"
D_A_REAL_LOSS_KEY = "d_real_a"
D_A_FAKE_LOSS_KEY = "d_fake_a"
D_B_REAL_LOSS_KEY = "d_real_b"
D_B_FAKE_LOSS_KEY = "d_fake_b"

# Set random seed for reproducibility
manualSeed = 999

# Number of training epochs
num_epochs = 200

#Running on local = 0, Running on COARE = 1, Running on CCS server = 2
server_config = 0
num_workers = 12

    