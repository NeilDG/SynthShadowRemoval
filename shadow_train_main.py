import itertools
import os
import sys
from optparse import OptionParser
import random
from pathlib import Path

import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import yaml
from torch import nn
from yaml.loader import SafeLoader
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from trainers import shadow_matte_trainer, shadow_removal_trainer
from tqdm import tqdm

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="VXX.XX")
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--save_per_iter', type=int, default=500)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--train_mode', type=str, default="all") #all, train_shadow_matte, train_shadow

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_to_load = opts.img_to_load
    global_config.train_mode = opts.train_mode

    config_holder = ConfigHolder.getInstance()
    network_config = config_holder.get_network_config()

    ## COARE - 24GB/P40
    if (global_config.server_config == 0):
        global_config.num_workers = 6
        global_config.disable_progress_bar = True #disable progress bar logging in COARE
        global_config.load_size = network_config["load_size"][0] - 4
        global_config.batch_size = network_config["batch_size"][0] - 4

        print("Using COARE configuration. Workers: ", global_config.num_workers)
        global_config.DATASET_PLACES_PATH = "/scratch3/neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/scratch3/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/scratch3/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "/scratch3/neil.delgallego/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "/scratch3/neil.delgallego/SRD_Test/srd/shadow_free/*.jpg"

    # CCS JUPYTER
    elif (global_config.server_config == 1):
        global_config.num_workers = 20
        global_config.load_size = network_config["load_size"][1]
        global_config.batch_size = network_config["batch_size"][1]
        global_config.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.DATASET_PLACES_PATH = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        global_config.ws_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "/home/jupyter-neil.delgallego/SRD_Train/shadow/*.jpg"
        global_config.ns_srd = "/home/jupyter-neil.delgallego/SRD_Train/shadow_free/*.jpg"

        print("Using CCS configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 2):
        global_config.num_workers = 6
        global_config.load_size = network_config["load_size"][2]
        global_config.batch_size = network_config["batch_size"][2]
        global_config.DATASET_PLACES_PATH = "X:/Datasets/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "X:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "X:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd ="X:/Datasets/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "X:/Datasets/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "X:/Datasets/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "X:/Datasets/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "X:/Datasets/SRD_Test/srd/shadow_free/*.jpg"

        print("Using HOME RTX2080Ti configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 3): #TITAN 3060
        global_config.num_workers = 4
        global_config.load_size = network_config["load_size"][2]
        global_config.batch_size = network_config["batch_size"][2]
        global_config.DATASET_PLACES_PATH = "/home/neildelgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/home/neildelgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/home/neildelgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd ="/home/neildelgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/home/neildelgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/home/neildelgallego/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "/home/neildelgallego/SRD_Train/shadow/*.jpg"
        global_config.ns_srd = "/home/neildelgallego/SRD_Train/shadow_free/*.jpg"

        print("Using TITAN configuration. Workers: ", global_config.num_workers)

    ## COARE - 40GB/A100
    elif (global_config.server_config == 4):
        global_config.num_workers = 12
        global_config.disable_progress_bar = True  # disable progress bar logging in COARE
        global_config.load_size = network_config["load_size"][1]
        global_config.batch_size = network_config["batch_size"][1]

        print("Using COARE configuration. Workers: ", global_config.num_workers)
        global_config.DATASET_PLACES_PATH = "/scratch3/neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/scratch3/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/scratch3/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/scratch3/neil.delgallego/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "/scratch3/neil.delgallego/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "/scratch3/neil.delgallego/SRD_Test/srd/shadow_free/*.jpg"
    else:
        global_config.num_workers = 12
        global_config.load_size = network_config["load_size"][0]
        global_config.batch_size = network_config["batch_size"][0]
        global_config.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "X:/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "X:/SRD_Test/srd/shadow_free/*.jpg"
        print("Using HOME RTX3090 configuration. Workers: ", global_config.num_workers)

def train_shadow(device, opts):
    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.ns_network_version = opts.network_version
    global_config.ns_iteration = opts.iteration
    global_config.test_size = 8

    tf = shadow_removal_trainer.ShadowTrainer(device)

    mode = "train_shadow"
    iteration = 0
    start_epoch = global_config.last_epoch_ns
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.ns_network_version, global_config.ns_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch_ns)
    print("---------------------------------------------------------------------------")

    # assert dataset_version == "v17", "Cannot identify dataset version."
    dataset_version = network_config["dataset_version"]
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)

    train_loader, dataset_count = dataset_loader.load_shadow_train_dataset()
    test_loader_train, _ = dataset_loader.load_shadow_test_dataset()

    if(dataset_version == "v_srd"):
        test_loader_istd, _ = dataset_loader.load_srd_dataset()
    else:
        test_loader_istd, _ = dataset_loader.load_istd_dataset()

    # compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    # plot utils
    # plot_loss_path = "./reports/train_test_loss.yaml"
    # l1_loss = nn.L1Loss()
    # if (os.path.exists(plot_loss_path)):
    #     with open(plot_loss_path) as f:
    #         losses_dict = yaml.load(f, SafeLoader)
    # else:
    #     losses_dict = {}
    #     losses_dict["train"] = []
    #     losses_dict["test_istd"] = []
    #
    # print("Losses dict: ", losses_dict["train"])
    # iteration = 7500

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader_istd))):
            _, rgb_ws, rgb_ns, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_matte = shadow_matte.to(device)

            _, rgb_ws_istd, rgb_ns_istd, matte_istd = test_data
            rgb_ws_istd = rgb_ws_istd.to(device)
            rgb_ns_istd = rgb_ns_istd.to(device)
            matte_istd = matte_istd.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_matte" : shadow_matte,
                         "rgb_ws_istd" : rgb_ws_istd, "rgb_ns_istd" : rgb_ns_istd, "matte_istd" : matte_istd}

            iteration = iteration + 1
            pbar.update(1)

            tf.train(epoch, iteration, input_map)

            if (iteration % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if (global_config.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "Train")

                    _, rgb_ws, rgb_ns, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_istd, "rgb_ns": rgb_ns_istd, "shadow_matte" : matte_istd}
                    tf.visdom_visualize(input_map, "Test ISTD")

        tf.save_states(epoch, iteration, True)
        if (tf.has_plateau(epoch)):
            break

            # if(global_config.plot_enabled == 1 and iteration % opts.save_per_iter == 0):
            #     rgb2ns_like = tf.test(input_map)
            #     train_loss = float(np.round(l1_loss(rgb2ns_like, rgb_ns).item(), 4))
            #     losses_dict["train"].append({iteration : train_loss})
            #
            #     input_map = {"rgb": rgb_ws_istd, "rgb_ns": rgb_ns_istd, "shadow_matte" : matte_istd}
            #     rgb2ns_like = tf.test(input_map)
            #     test_loss = float(np.round(l1_loss(rgb2ns_like, rgb_ns_istd).item(), 4))
            #     losses_dict["test_istd"].append({iteration: test_loss})
            #
            #     plot_loss_file = open(plot_loss_path, "w")
            #     yaml.dump(losses_dict, plot_loss_file)
            #     plot_loss_file.close()
            #     print("Dumped train test loss to ", plot_loss_path)

    pbar.close()

def train_shadow_matte(device, opts):
    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.sm_network_version = opts.network_version
    global_config.sm_iteration = opts.iteration
    global_config.test_size = 8

    tf = shadow_matte_trainer.ShadowMatteTrainer(device)

    mode = "train_shadow_matte"
    iteration = 0
    start_epoch = global_config.last_epoch_sm
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.sm_network_version, global_config.sm_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch_sm)
    print("---------------------------------------------------------------------------")

    dataset_version = network_config["dataset_version"]
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)

    dataset_val = ConfigHolder.getInstance().get_network_attribute("dataset_val_ref", "istd+srd")

    train_loader_synth, dataset_count = dataset_loader.load_shadow_train_dataset()
    test_loader_train, _ = dataset_loader.load_shadow_test_dataset()
    if(dataset_val == "istd+srd"):
        print("Loading ISTD + SRD as validation dataset")
        test_loader, _ = dataset_loader.load_istd_srd_dataset()
    elif (dataset_val == "istd"):
        print("Loading ISTD as validation dataset")
        test_loader, _ = dataset_loader.load_istd_dataset()
    else:
        print("Loading SRD as validation dataset")
        test_loader, _ = dataset_loader.load_srd_dataset()

    #compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, max_epochs):
        for i, (train_data, test_data) in enumerate(zip(train_loader_synth, itertools.cycle(test_loader))):
            _, rgb_ws, rgb_ns, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_matte = shadow_matte.to(device)

            _, rgb_ws_val, rgb_ns_val, matte_val = test_data
            rgb_ws_val = rgb_ws_val.to(device)
            rgb_ns_val = rgb_ns_val.to(device)
            matte_val = matte_val.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_matte": shadow_matte,
                         "rgb_ws_val": rgb_ws_val, "matte_val": matte_val}

            tf.train(epoch, iteration, input_map)
            iteration = iteration + 1
            pbar.update(1)

            if (iteration % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "Train")

                    _, rgb_ws, rgb_ns, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_val, "rgb_ns" : rgb_ns_val, "shadow_matte": matte_val}
                    tf.visdom_visualize(input_map, "Test ISTD")

        tf.save_states(epoch, iteration, True)
        if(tf.has_plateau(epoch)):
            break

    pbar.close()

def prepare_training():
    NETWORK_SAVE_PATH = "./checkpoint/by_epoch/"
    try:
        path = Path(NETWORK_SAVE_PATH)
        path.mkdir(parents=True)
    except OSError as error:
        print(NETWORK_SAVE_PATH + " already exists. Skipping.", error)

    NETWORK_SAVE_PATH = "./checkpoint/by_sample/"
    try:
        path = Path(NETWORK_SAVE_PATH)
        path.mkdir(parents=True)
    except OSError as error:
        print(NETWORK_SAVE_PATH + " already exists. Skipping.", error)

    BEST_NETWORK_SAVE_PATH = "./checkpoint/best/"
    try:
        path = Path(BEST_NETWORK_SAVE_PATH)
        path.mkdir(parents=True)
    except OSError as error:
        print(BEST_NETWORK_SAVE_PATH + " already exists. Skipping.", error)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    assert opts.train_mode == "train_shadow" or opts.train_mode == "train_shadow_matte", "Unrecognized train mode: " + opts.train_mode
    plot_utils.VisdomReporter.initialize()

    prepare_training()

    if (opts.train_mode == "train_shadow_matte"):
        train_shadow_matte(device, opts)
    elif(opts.train_mode == "train_shadow"):
        train_shadow(device, opts)


if __name__ == "__main__":
    main(sys.argv)
