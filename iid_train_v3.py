import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from config import iid_server_config
from loaders import dataset_loader
from model.modules import shadow_matte_pool
from trainers import iid_trainer
from trainers import early_stopper
from transforms import iid_transforms
import constants
from utils import plot_utils
from trainers import trainer_factory
from tqdm import tqdm
from tqdm.auto import trange
from time import sleep

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--shadow_matte_network_version', type=str, default="VXX.XX")
parser.add_option('--shadow_removal_version', type=str, default="VXX.XX")
parser.add_option('--shadow_matte_iteration', type=int, default="1")
parser.add_option('--shadow_removal_iteration', type=int, default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0005")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--debug_run', type=int, help="Debug mode?", default=0)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--train_mode', type=str, default="all") #all, train_shadow_matte, train_shadow

def update_config(opts):
    constants.server_config = opts.server_config
    constants.plot_enabled = opts.plot_enabled
    constants.debug_run = opts.debug_run

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_B/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.DATASET_PLACES_PATH = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        constants.ws_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_B/*.png"

        print("Using CCS configuration. Workers: ", opts.num_workers)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/home/neil_delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "/home/neil_delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd = "/home/neil_delgallego/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "/home/neil_delgallego/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "/home/neil_delgallego/ISTD_Dataset/test/test_B/*.png"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "C:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd ="C:/Datasets/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "C:/Datasets/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "C:/Datasets/ISTD_Dataset/test/test_B/*.png"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        constants.ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"
        constants.mask_istd = "E:/ISTD_Dataset/test/test_B/*.png"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers)

def train_shadow(tf, device, opts):
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_shadow_network_params_from_version()
    print("General config:", general_config)
    print("Network config: ", network_config)

    mode = "train_shadow"
    patch_size = general_config[mode]["patch_size"]
    dataset_version = network_config["dataset_version"]

    # assert dataset_version == "v17", "Cannot identify dataset version."
    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    load_size = network_config["load_size_z"]

    train_loader, dataset_count = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, patch_size, load_size, opts)
    test_loader_train, _ = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    test_loader_istd, _ = dataset_loader.load_istd_dataset(constants.ws_istd, constants.ns_istd, constants.mask_istd, load_size, opts)

    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode(mode)
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    needed_progress = int((general_config[mode]["max_epochs"]) * (dataset_count / load_size))
    current_progress = int(start_epoch * (dataset_count / load_size))
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader_istd))):
            _, rgb_ws, rgb_ns, rgb_ws_gray, shadow_map, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            rgb_ws_gray = rgb_ws_gray.to(device)
            shadow_map = shadow_map.to(device)
            shadow_matte = shadow_matte.to(device)

            _, rgb_ws_istd, rgb_ns_istd, gray_istd, matte_istd = test_data
            rgb_ws_istd = rgb_ws_istd.to(device)
            rgb_ns_istd = rgb_ns_istd.to(device)
            gray_istd = gray_istd.to(device)
            matte_istd = matte_istd.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "rgb_ws_gray" : rgb_ws_gray, "shadow_map" : shadow_map, "shadow_matte" : shadow_matte,
                         "rgb_ws_istd" : rgb_ws_istd, "rgb_ns_istd" : rgb_ns_istd, "gray_istd" : gray_istd, "matte_istd" : matte_istd}
            target_map = input_map

            tf.train(mode, epoch, iteration, input_map, target_map)
            iteration = iteration + 1
            pbar.update(1)

            if (tf.is_stop_condition_met(mode)):
                break

            if (i % 300 == 0):
                tf.save(mode, epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(mode, iteration)
                    tf.visdom_visualize(mode, input_map, "Train")

                    _, rgb_ws, rgb_ns, rgb_ws_gray, shadow_map, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)
                    rgb_ws_gray = rgb_ws_gray.to(device)
                    shadow_map = shadow_map.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "rgb_ws_gray" : rgb_ws_gray, "shadow_map": shadow_map, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(mode, input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_istd, "rgb_ns": rgb_ns_istd, "rgb_ws_gray" : gray_istd, "shadow_matte" : matte_istd}
                    tf.visdom_visualize(mode, input_map, "Test ISTD")

        if (tf.is_stop_condition_met(mode)):
            break

    pbar.close()

def train_shadow_matte(tf, device, opts):
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_shadow_matte_params_from_version()
    print("General config:", general_config)
    print("Network config: ", network_config)

    mode = "train_shadow_matte"
    patch_size = general_config[mode]["patch_size"]
    dataset_version = network_config["dataset_version"]

    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    load_size = network_config["load_size_m"]

    train_loader_synth, dataset_count = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, patch_size, load_size, opts)
    train_loader_istd, _ = dataset_loader.load_istd_train_dataset(constants.ws_istd, constants.ns_istd, patch_size, load_size, opts)
    test_loader_train, _ = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    test_loader_istd, _ = dataset_loader.load_istd_dataset(constants.ws_istd, constants.ns_istd, constants.mask_istd, load_size, opts)

    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode(mode)
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    #compute total progress
    needed_progress = int((general_config[mode]["max_epochs"]) * (dataset_count / load_size))
    current_progress = int(start_epoch * (dataset_count / load_size))
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
        for i, (train_data, train_data_istd, test_data) in enumerate(zip(train_loader_synth, itertools.cycle(train_loader_istd), itertools.cycle(test_loader_istd))):
            _, rgb_ws, rgb_ns, rgb_ws_gray, shadow_map, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            rgb_ws_gray = rgb_ws_gray.to(device)
            shadow_map = shadow_map.to(device)
            shadow_matte = shadow_matte.to(device)

            _, _, _, _, _, matte_train_istd = train_data_istd
            matte_train_istd = matte_train_istd.to(device)

            _, rgb_ws_istd, rgb_ns_istd, gray_istd, matte_istd = test_data
            rgb_ws_istd = rgb_ws_istd.to(device)
            rgb_ns_istd = rgb_ns_istd.to(device)
            gray_istd = gray_istd.to(device)
            matte_istd = matte_istd.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "rgb_ws_gray": rgb_ws_gray, "shadow_map": shadow_map, "shadow_matte": shadow_matte,
                         "rgb_ws_istd": rgb_ws_istd, "rgb_ns_istd": rgb_ns_istd, "gray_istd": gray_istd, "matte_istd": matte_istd,
                         "matte_train_istd" : matte_train_istd}
            target_map = input_map

            tf.train(mode, epoch, iteration, input_map, target_map)
            iteration = iteration + 1
            pbar.update(1)

            if (tf.is_stop_condition_met(mode)):
                break

            if ((iteration - 1) % 150 == 0):
                tf.save(mode, epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(mode, iteration)
                    tf.visdom_visualize(mode, input_map, "Train")

                    _, rgb_ws, rgb_ns, rgb_ws_gray, shadow_map, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ws_gray = rgb_ws_gray.to(device)
                    rgb_ns = rgb_ns.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "rgb_ws_gray": rgb_ws_gray, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(mode, input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_istd, "rgb_ns" : rgb_ns_istd,  "rgb_ws_gray": gray_istd, "shadow_matte": matte_istd}
                    tf.visdom_visualize(mode, input_map, "Test ISTD")

        if (tf.is_stop_condition_met(mode)):
            break

    pbar.close()

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
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    plot_utils.VisdomReporter.initialize()

    constants.shadow_removal_version = opts.shadow_removal_version
    constants.shadow_matte_network_version = opts.shadow_matte_network_version
    iid_server_config.IIDServerConfig.initialize()

    tf = trainer_factory.TrainerFactory(device, opts)
    tf.initialize_all_trainers(opts)

    assert opts.train_mode == "all" or opts.train_mode == "train_shadow" or opts.train_mode == "train_shadow_matte", "Unrecognized train mode: " + opts.train_mode

    if(opts.train_mode == "all"):
        train_shadow_matte(tf, device, opts)
        train_shadow(tf, device, opts)
    elif (opts.train_mode == "train_shadow_matte"):
        train_shadow_matte(tf, device, opts)
    elif(opts.train_mode == "train_shadow"):
        train_shadow(tf, device, opts)


if __name__ == "__main__":
    main(sys.argv)
