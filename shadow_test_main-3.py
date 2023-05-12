###
# For testing of GTA, CGI, and IIW test dataset performance
###

import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import yaml
from yaml import SafeLoader
from config.network_config import ConfigHolder
from testers.shadow_tester_class import TesterClass
from loaders import dataset_loader
import global_config
from utils import plot_utils
from tqdm import tqdm
from trainers import shadow_matte_trainer, shadow_removal_trainer

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--img_vis_enabled', type=int, default=1)
parser.add_option('--shadow_matte_version', type=str, default="VXX.XX")
parser.add_option('--shadow_removal_version', type=str, default="VXX.XX")
parser.add_option('--shadow_matte_iteration', type=int, default="1")
parser.add_option('--shadow_removal_iteration', type=int, default="1")
parser.add_option('--train_mode', type=str, default="all") #all, train_shadow_matte, train_shadow
parser.add_option('--dataset_target', type=str, default="all") #all, train, istd, srd, usr

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.img_to_load = opts.img_to_load
    global_config.dataset_target = opts.dataset_target
    global_config.num_workers = 12
    global_config.test_size = 128
    global_config.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
    global_config.rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    global_config.rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    print("Using HOME RTX3090 configuration. Workers: ", global_config.num_workers)

def test_shadow_matte(dataset_tester, opts):
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")

    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=global_config.sm_network_config["dataset_version"])
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=global_config.sm_network_config["dataset_version"])
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)

    if(global_config.dataset_target == "all" or global_config.dataset_target == "train"):
        shadow_loader, dataset_count = dataset_loader.load_shadow_test_dataset()
        needed_progress = int(dataset_count / global_config.test_size) + 1
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        for i, (file_name, rgb_ws, _, _, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws = rgb_ws.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_shadow_matte(file_name, rgb_ws, shadow_matte, "Train", opts.img_vis_enabled, False, opts)
            pbar.update(1)
            if (i % 16 == 0):
                break

        dataset_tester.print_shadow_matte_performance("SM - Train Set")

    # ISTD test dataset
    if (global_config.dataset_target == "all" or global_config.dataset_target == "istd"):
        shadow_loader, dataset_count = dataset_loader.load_istd_dataset()
        needed_progress = int(dataset_count / global_config.test_size) + 1
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        for i, (file_name, rgb_ws, _, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws = rgb_ws.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_shadow_matte(file_name, rgb_ws, shadow_matte, "ISTD", opts.img_vis_enabled, True, opts)
            pbar.update(1)
            # break

        dataset_tester.print_shadow_matte_performance("SM - ISTD")

    #SRD test dataset
    if (global_config.dataset_target == "all" or global_config.dataset_target == "srd"):
        shadow_loader, dataset_count = dataset_loader.load_srd_dataset()
        needed_progress = int(dataset_count / global_config.test_size) + 1
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        for i, (file_name, rgb_ws, _, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws = rgb_ws.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_shadow_matte(file_name, rgb_ws, shadow_matte, "SRD", opts.img_vis_enabled, True, opts)
            pbar.update(1)
            # break

        dataset_tester.print_shadow_matte_performance("SM - SRD")

    if (global_config.dataset_target == "all" or global_config.dataset_target == "usr"):
        shadow_loader, dataset_count = dataset_loader.load_usr_dataset()
        needed_progress = int(dataset_count / global_config.test_size) + 1
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        for i, (file_name, rgb_ws) in enumerate(shadow_loader, 0):
            rgb_ws = rgb_ws.to(device)

            dataset_tester.test_shadow_matte_usr(file_name, rgb_ws, "USR", opts.img_vis_enabled, True)
            pbar.update(1)
            # break



def test_shadow_removal(dataset_tester, opts):
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")

    update_config(opts)
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=global_config.ns_network_config["dataset_version"])
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=global_config.ns_network_config["dataset_version"])
    print("NS shadow removal DS version: ", global_config.ns_network_config["dataset_version"])
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)

    # SHADOW dataset test
    # Using train dataset
    if (global_config.dataset_target == "all" or global_config.dataset_target == "train"):
        shadow_loader, dataset_count = dataset_loader.load_shadow_test_dataset()
        needed_progress = int(dataset_count / global_config.test_size)
        pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
        for i, (_, rgb_ws, rgb_ns, _, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_shadow(rgb_ws, rgb_ns, shadow_matte, "Train", opts.img_vis_enabled, opts.train_mode)
            pbar.update(1)
            if ((i + 1) % 4 == 0):
                break

        dataset_tester.print_ave_shadow_performance("Train Set")

    # ISTD test dataset
    if (global_config.dataset_target == "all" or global_config.dataset_target == "istd"):
        shadow_loader, _ = dataset_loader.load_istd_dataset()
        for i, (file_name, rgb_ws, rgb_ns, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws_tensor = rgb_ws.to(device)
            rgb_ns_tensor = rgb_ns.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_istd_shadow(file_name, rgb_ws_tensor, rgb_ns_tensor, shadow_matte, opts.img_vis_enabled, 1, opts.train_mode)
            # break

        dataset_tester.print_ave_shadow_performance("ISTD")

    # SRD test dataset
    if (global_config.dataset_target == "all" or global_config.dataset_target == "srd"):
        shadow_loader, _ = dataset_loader.load_srd_dataset()
        for i, (file_name, rgb_ws, rgb_ns, shadow_matte) in enumerate(shadow_loader, 0):
            rgb_ws_tensor = rgb_ws.to(device)
            rgb_ns_tensor = rgb_ns.to(device)
            shadow_matte = shadow_matte.to(device)

            dataset_tester.test_srd(file_name, rgb_ws_tensor, rgb_ns_tensor, shadow_matte, opts.img_vis_enabled, 1, opts.train_mode)
            # break

        dataset_tester.print_ave_shadow_performance("SRD")

    if (global_config.dataset_target == "all" or global_config.dataset_target == "usr"):
        shadow_loader, _ = dataset_loader.load_usr_dataset()
        for i, (file_name, rgb_ws) in enumerate(shadow_loader, 0):
            rgb_ws_tensor = rgb_ws.to(device)

            dataset_tester.test_usr(file_name, rgb_ws_tensor, opts.img_vis_enabled, 1)
            # break

    # PLACES test dataset
    # shadow_loader = dataset_loader.load_single_test_dataset(constants.imgx_dir, opts)  # load PLACES
    # for i, (file_name, rgb_ws) in enumerate(shadow_loader, 0):
    #     rgb_ws = rgb_ws.to(device)
    #     dataset_tester.test_any_image(file_name, rgb_ws, "Places-365", opts.img_vis_enabled, 1, opts)


def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    plot_utils.VisdomReporter.initialize()

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.shadow_matte_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    global_config.sm_network_version = opts.shadow_matte_version
    global_config.sm_iteration = opts.shadow_matte_iteration
    global_config.sm_network_config = ConfigHolder.getInstance().get_network_config()
    shadow_m = shadow_matte_trainer.ShadowMatteTrainer(device)

    print("---------------------------------------------------------------------------")
    print("Successfully loaded shadow matte network: ", opts.shadow_matte_version, str(opts.shadow_matte_iteration))
    print("Network config: ", global_config.sm_network_config)
    print("---------------------------------------------------------------------------")

    ConfigHolder.destroy()  # for security, destroy config holder since it should no longer be needed

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.shadow_removal_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    global_config.ns_network_version = opts.shadow_removal_version
    global_config.ns_iteration = opts.shadow_removal_iteration
    global_config.ns_network_config = ConfigHolder.getInstance().get_network_config()
    global_config.load_per_sample = True
    shadow_t = shadow_removal_trainer.ShadowTrainer(device)
    shadow_t.load_state_for_specific_sample()

    print("---------------------------------------------------------------------------")
    print("Successfully loaded shadow removal network: ", opts.shadow_removal_version, str(opts.shadow_removal_iteration))
    print("Network config: ", global_config.ns_network_config)
    print("---------------------------------------------------------------------------")

    # ConfigHolder.destroy() #for security, destroy config holder since it should no longer be needed

    dataset_tester = TesterClass(shadow_m, shadow_t)

    if(opts.train_mode == "train_shadow_matte"):
        test_shadow_matte(dataset_tester, opts)
    elif(opts.train_mode == "train_shadow"):
        test_shadow_removal(dataset_tester, opts)
    else:
        # test_shadow_matte(dataset_tester, opts)
        test_shadow_removal(dataset_tester, opts)


if __name__ == "__main__":
    main(sys.argv)