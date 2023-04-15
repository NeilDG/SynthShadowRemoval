import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import yaml
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

    ## COARE
    if (global_config.server_config == 1):
        global_config.num_workers = 6
        global_config.disable_progress_bar = True #disable progress bar logging in COARE
        global_config.load_size = network_config["load_size"][0]
        global_config.batch_size = network_config["batch_size"][0]

        print("Using COARE configuration. Workers: ", global_config.num_workers)
        global_config.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "/scratch1/scratch2/neil.delgallego/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "/scratch1/scratch2/neil.delgallego/SRD_Test/srd/shadow_free/*.jpg"

    # CCS JUPYTER
    elif (global_config.server_config == 2):
        global_config.num_workers = 20
        global_config.load_size = network_config["load_size"][1]
        global_config.batch_size = network_config["batch_size"][1]
        global_config.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.DATASET_PLACES_PATH = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        global_config.ws_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "/home/jupyter-neil.delgallego/ISTD_Dataset/test/test_B/*.png"

        print("Using CCS configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 3):
        global_config.num_workers = 6
        global_config.load_size = network_config["load_size"][2]
        global_config.batch_size = network_config["batch_size"][2]
        global_config.DATASET_PLACES_PATH = "C:/Datasets/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
        global_config.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
        global_config.ws_istd ="C:/Datasets/ISTD_Dataset/test/test_A/*.png"
        global_config.ns_istd = "C:/Datasets/ISTD_Dataset/test/test_C/*.png"
        global_config.mask_istd = "C:/Datasets/ISTD_Dataset/test/test_B/*.png"
        global_config.ws_srd = "C:/Datasets/SRD_Test/srd/shadow/*.jpg"
        global_config.ns_srd = "C:/Datasets/SRD_Test/srd/shadow_free/*.jpg"

        print("Using HOME RTX2080Ti configuration. Workers: ", global_config.num_workers)

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

    tf = shadow_removal_trainer.ShadowTrainer(device)

    mode = "train_shadow"
    iteration = 0
    start_epoch = global_config.last_epoch
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.ns_network_version, global_config.ns_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch)
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
        test_loader_istd, _ = dataset_loader.load_istd_dataset()
    else:
        test_loader_istd, _ = dataset_loader.load_istd_dataset()

    # compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader_istd))):
            _, rgb_ws, rgb_ns, shadow_map, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_map = shadow_map.to(device)
            shadow_matte = shadow_matte.to(device)

            _, rgb_ws_istd, rgb_ns_istd, matte_istd = test_data
            rgb_ws_istd = rgb_ws_istd.to(device)
            rgb_ns_istd = rgb_ns_istd.to(device)
            matte_istd = matte_istd.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns , "shadow_map" : shadow_map, "shadow_matte" : shadow_matte,
                         "rgb_ws_istd" : rgb_ws_istd, "rgb_ns_istd" : rgb_ns_istd, "matte_istd" : matte_istd}
            target_map = input_map

            iteration = iteration + 1
            pbar.update(1)

            tf.train(epoch, iteration, input_map, target_map)
            if (tf.is_stop_condition_met()):
                break

            if (i % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "Train")

                    _, rgb_ws, rgb_ns, shadow_map, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)
                    shadow_map = shadow_map.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns , "shadow_map": shadow_map, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_istd, "rgb_ns": rgb_ns_istd, "shadow_matte" : matte_istd}
                    tf.visdom_visualize(input_map, "Test ISTD")


        if (tf.is_stop_condition_met()):
            break

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

    train_loader_synth, dataset_count = dataset_loader.load_shadow_train_dataset()
    test_loader_train, _ = dataset_loader.load_shadow_test_dataset()
    if (dataset_version == "v_srd"):
        test_loader_istd, _ = dataset_loader.load_istd_dataset()
    else:
        test_loader_istd, _ = dataset_loader.load_istd_dataset()

    #compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, max_epochs):
        for i, (train_data, test_data) in enumerate(zip(train_loader_synth, itertools.cycle(test_loader_istd))):
            _, rgb_ws, rgb_ns, shadow_map, shadow_matte = train_data
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_map = shadow_map.to(device)
            shadow_matte = shadow_matte.to(device)

            _, rgb_ws_istd, rgb_ns_istd, matte_istd = test_data
            rgb_ws_istd = rgb_ws_istd.to(device)
            matte_istd = matte_istd.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_map": shadow_map, "shadow_matte": shadow_matte,
                         "rgb_ws_istd": rgb_ws_istd, "rgb_ns_istd": rgb_ns_istd, "matte_istd": matte_istd}
            target_map = input_map

            tf.train(epoch, iteration, input_map, target_map)

            if (tf.is_stop_condition_met()):
                break

            if (iteration % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "Train")

                    _, rgb_ws, rgb_ns, shadow_map, shadow_matte = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)
                    shadow_matte = shadow_matte.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_matte": shadow_matte}
                    tf.visdom_visualize(input_map, "Test Synthetic")

                    input_map = {"rgb": rgb_ws_istd, "rgb_ns" : rgb_ns_istd, "shadow_matte": matte_istd}
                    tf.visdom_visualize(input_map, "Test ISTD")

            iteration = iteration + 1
            pbar.update(1)

        if (tf.is_stop_condition_met()):
            break

    pbar.close()

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


    if (opts.train_mode == "train_shadow_matte"):
        train_shadow_matte(device, opts)
    elif(opts.train_mode == "train_shadow"):
        train_shadow(device, opts)


if __name__ == "__main__":
    main(sys.argv)
