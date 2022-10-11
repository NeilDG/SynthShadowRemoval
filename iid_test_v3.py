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
parser.add_option('--version', type=str, default="")
parser.add_option('--iteration', type=int, default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--img_vis_enabled', type=int, default=1)
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
    constants.ITERATION = str(opts.iteration)
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

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "E:/SynthWeather Dataset 8/unlit/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers)

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

    constants.network_version = opts.version
    iid_server_config.IIDServerConfig.initialize()
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    network_config = sc_instance.interpret_network_config_from_version()
    refine_enabled = False

    # version_z = opts.version
    # iteration_z = opts.iteration
    #
    # opts.version = version_a
    # opts.iteration = iteration_a
    # sc_instance.update_version_config()
    # tf = trainer_factory.TrainerFactory(device, opts)
    # albedo_t = tf.get_albedo_trainer()

    # opts.version = version_s
    # opts.iteration = iteration_s
    # sc_instance.update_version_config()
    # tf = trainer_factory.TrainerFactory(device, opts)
    # shading_t = tf.get_shading_trainer()

    # opts.version = version_z
    # opts.iteration = iteration_z
    sc_instance.update_version_config()
    tf = trainer_factory.TrainerFactory(device, opts)
    shadow_p = tf.get_shadow_mask_trainer()
    shadow_t = tf.get_shadow_trainer()
    shadow_rt = tf.get_shadow_refine_trainer()

    dataset_tester = TesterClass(shadow_p, shadow_t, shadow_rt)

    dataset_version = network_config["dataset_version"]

    assert dataset_version == "v8" or dataset_version == "v15", "Cannot identify dataset version."
    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    print("Dataset path: ", rgb_dir_ws, rgb_dir_ns)

    #SHADOW dataset test
    #Using train dataset
    print(rgb_dir_ws, rgb_dir_ns)
    shadow_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, 256, 8, opts)
    for i, (file_name, rgb_ws, rgb_ns, shadow_map, shadow_mask) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)
        rgb_ns = rgb_ns.to(device)
        shadow_map = shadow_map.to(device)
        shadow_mask = shadow_mask.to(device)

        dataset_tester.test_shadow(rgb_ws, rgb_ns, shadow_map, shadow_mask, "Train",  refine_enabled, opts.img_vis_enabled, opts)
        if (i % 16 == 0):
            break

    dataset_tester.print_ave_shadow_performance("Train Set", opts)

    # ISTD test dataset
    shadow_loader = dataset_loader.load_istd_dataset(constants.ws_istd, constants.ns_istd, constants.mask_istd, 8, opts)
    for i, (file_name, rgb_ws, rgb_ns, shadow_mask) in enumerate(shadow_loader, 0):
        rgb_ws_tensor = rgb_ws.to(device)
        rgb_ns_tensor = rgb_ns.to(device)
        shadow_mask = shadow_mask.to(device)

        dataset_tester.test_istd_shadow(file_name, rgb_ws_tensor, rgb_ns_tensor, shadow_mask, opts.img_vis_enabled, 1, 0,  opts)
        # break

    dataset_tester.print_ave_shadow_performance("ISTD", opts)

    # cgi_rgb_dir = "E:/CGIntrinsics/images/*/*_mlt.png"
    # rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)
    #
    # test_loader = dataset_loader.load_cgi_dataset(cgi_rgb_dir, 480, opts)
    # for i, (file_name, rgb_batch, albedo_batch) in enumerate(test_loader, 0):
    #     # CGI dataset
    #     rgb_tensor = rgb_batch.to(device)
    #     albedo_tensor = albedo_batch.to(device)
    #     dataset_tester.test_cgi(rgb_tensor, albedo_tensor, opts)
    #     break
    #
    # #IIW dataset
    # iiw_rgb_dir = "E:/iiw-decompositions/original_image/*.jpg"
    # test_loader = dataset_loader.load_iiw_dataset(iiw_rgb_dir, opts)
    # for i, (file_name, rgb_img) in enumerate(test_loader, 0):
    #     with torch.no_grad():
    #         rgb_tensor = rgb_img.to(device)
    #         dataset_tester.test_iiw(file_name, rgb_tensor, opts)
    #
    # dataset_tester.get_average_whdr(opts)
    #
    # #check RW performance
    # _, input_rgb_batch = next(iter(rw_loader))
    # input_rgb_tensor = input_rgb_batch.to(device)
    # dataset_tester.test_rw(input_rgb_tensor, opts)
    #
    # #measure GTA performance
    # dataset_tester.test_gta(opts)
    # iid_test_v2.measure_performance(opts)

if __name__ == "__main__":
    main(sys.argv)