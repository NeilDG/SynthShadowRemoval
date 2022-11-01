###
# For testing of GTA, CGI, and IIW test dataset performance
###

import glob
import sys
from optparse import OptionParser
import random
import cv2
import kornia
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import iid_test_v2
from config import iid_server_config
from iid_test_v2 import TesterClass
from loaders import dataset_loader
from transforms import iid_transforms
import constants
from utils import plot_utils, tensor_utils
from trainers import trainer_factory
from custom_losses import whdr

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--shadow_matte_network_version', type=str, default="VXX.XX")
parser.add_option('--shadow_removal_version', type=str, default="VXX.XX")
parser.add_option('--shadow_matte_iteration', type=int, default="1")
parser.add_option('--shadow_removal_iteration', type=int, default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--img_vis_enabled', type=int, default=1)
parser.add_option('--debug_policy', type=int, default=0)
parser.add_option('--input_path', type=str)
parser.add_option('--output_path', type=str)
parser.add_option('--img_size', type=int, default=(256, 256))

# version_a ="v13.07"
# iteration_a = 8
# version_s = "v12.07"
# iteration_s = 15
# version_z = "v16.10"
# iteration_z = 10

def update_config(opts):
    constants.server_config = opts.server_config
    constants.plot_enabled = opts.plot_enabled

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_C/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.DATASET_PLACES_PATH = constants.rgb_dir_ws
        constants.ws_istd = constants.rgb_dir_ws
        constants.ns_istd = constants.rgb_dir_ns

        print("Using CCS configuration. Workers: ", opts.num_workers)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "/home/neil_delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "C:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd = "C:/Datasets/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "C:/Datasets/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "C:/Datasets/ISTD_Dataset/test/test_B/*.png"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "E:/SynthWeather Dataset 8/unlit/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers)

def test_shadow_matte(dataset_tester, opts):
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")

    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    sc_instance.update_version_config()
    network_config = sc_instance.interpret_shadow_matte_params_from_version()
    general_config = sc_instance.get_general_configs()
    print("General config:", general_config)
    print("Network config: ", network_config)

    dataset_version = network_config["dataset_version"]

    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    print("Dataset path: ", rgb_dir_ws, rgb_dir_ns)
    shadow_loader, _ = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    for i, (_, rgb_ws, _, rgb_ws_gray, _, shadow_matte) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)
        rgb_ws_gray = rgb_ws_gray.to(device)
        shadow_matte = shadow_matte.to(device)

        dataset_tester.test_shadow_matte(rgb_ws, rgb_ws_gray, shadow_matte, "Train", opts.img_vis_enabled, opts)
        if (i % 16 == 0):
            break

    dataset_tester.print_shadow_matte_performance("SM - Train Set", opts)

    # ISTD test dataset
    shadow_loader, _ = dataset_loader.load_istd_dataset(constants.ws_istd, constants.ns_istd, constants.mask_istd, 8, opts)
    for i, (_, rgb_ws, _, rgb_ws_gray, shadow_matte) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)
        rgb_ws_gray = rgb_ws_gray.to(device)
        shadow_matte = shadow_matte.to(device)

        dataset_tester.test_shadow_matte(rgb_ws, rgb_ws_gray, shadow_matte, "ISTD", opts.img_vis_enabled, opts)
        # break

    dataset_tester.print_shadow_matte_performance("SM - ISTD", opts)

def test_shadow_removal(dataset_tester, opts):
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")

    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    network_config = sc_instance.interpret_shadow_matte_params_from_version()
    sc_instance.update_version_config()

    dataset_version = network_config["dataset_version"]

    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    print("Dataset path: ", rgb_dir_ws, rgb_dir_ns)
    # SHADOW dataset test
    # Using train dataset
    shadow_loader, _ = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    for i, (_, rgb_ws, rgb_ns, _, _, shadow_matte) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)
        rgb_ns = rgb_ns.to(device)
        shadow_matte = shadow_matte.to(device)

        dataset_tester.test_shadow(rgb_ws, rgb_ns, shadow_matte, "Train", opts.img_vis_enabled, opts.debug_policy, opts)
        if (i % 16 == 0):
            break

    dataset_tester.print_ave_shadow_performance("Train Set", opts)

    # SRD test dataset
    shadow_loader, _ = dataset_loader.load_srd_dataset(constants.ws_srd, constants.ns_srd, 8, opts)
    for i, (file_name, rgb_ws, rgb_ns, _, shadow_matte) in enumerate(shadow_loader, 0):
        rgb_ws_tensor = rgb_ws.to(device)
        rgb_ns_tensor = rgb_ns.to(device)
        shadow_matte = shadow_matte.to(device)

        dataset_tester.test_srd(file_name, rgb_ws_tensor, rgb_ns_tensor, shadow_matte, opts.img_vis_enabled, 1, opts.debug_policy, opts)
        # break

    dataset_tester.print_ave_shadow_performance("SRD", opts)

    # ISTD test dataset
    shadow_loader, _ = dataset_loader.load_istd_dataset(constants.ws_istd, constants.ns_istd, constants.mask_istd, 8, opts)
    for i, (file_name, rgb_ws, rgb_ns, _, shadow_matte) in enumerate(shadow_loader, 0):
        rgb_ws_tensor = rgb_ws.to(device)
        rgb_ns_tensor = rgb_ns.to(device)
        shadow_matte = shadow_matte.to(device)

        dataset_tester.test_istd_shadow(file_name, rgb_ws_tensor, rgb_ns_tensor, shadow_matte, opts.img_vis_enabled, 1, opts.debug_policy, opts)
        # break

    dataset_tester.print_ave_shadow_performance("ISTD", opts)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    print(constants.rgb_dir_ws_styled, constants.albedo_dir)
    plot_utils.VisdomReporter.initialize()

    constants.shadow_matte_network_version = opts.shadow_matte_network_version
    constants.shadow_removal_version = opts.shadow_removal_version

    iid_server_config.IIDServerConfig.initialize()
    tf = trainer_factory.TrainerFactory(device, opts)
    shadow_m = tf.get_shadow_matte_trainer()
    shadow_t = tf.get_shadow_trainer()

    dataset_tester = TesterClass(shadow_m, shadow_t)
    test_shadow_matte(dataset_tester, opts)
    # test_shadow_removal(dataset_tester, opts)


if __name__ == "__main__":
    main(sys.argv)