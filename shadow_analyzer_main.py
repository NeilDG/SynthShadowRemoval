###
# For analyzing of shadow removal datasets
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
from utils import plot_utils, tensor_utils
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
parser.add_option('--load_best', type=int, default=0)
parser.add_option('--train_mode', type=str, default="all") #all, train_shadow_matte, train_shadow
parser.add_option('--dataset_target', type=str, default="all") #all, train, istd, srd, usr

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.img_to_load = opts.img_to_load
    global_config.dataset_target = opts.dataset_target
    global_config.num_workers = 12
    global_config.test_size = 64
    global_config.load_best = bool(opts.load_best)
    global_config.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
    global_config.rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    global_config.rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    print("Using HOME RTX3090 configuration. Workers: ", global_config.num_workers)

def analyze_luminance():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    shadow_loader, dataset_count = dataset_loader.load_istd_dataset()
    needed_progress = int(dataset_count / global_config.test_size) + 1
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)

    luminance_min_total = []
    luminance_max_total = []
    luminance_mean_total = []

    for i, (file_name, rgb_ws, _, shadow_mask, _) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)
        shadow_mask = shadow_mask.to(device)

        pbar.update(1)
        luminance_map = tensor_utils.get_luminance(rgb_ws * shadow_mask)
        visdom_reporter.plot_image(rgb_ws, "ISTD WS Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
        visdom_reporter.plot_image(luminance_map, "ISTD Luminance Map - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))

        luminance_vector = torch.flatten(luminance_map)
        luminance_min = np.round(torch.min(luminance_vector[luminance_vector > 0.0]).item(), 4)
        luminance_max = np.round(torch.max(luminance_vector[luminance_vector > 0.0]).item(), 4)
        luminance_mean = np.round(torch.mean(luminance_vector[luminance_vector > 0.0]).item(), 4)

        luminance_min_total.append(luminance_min)
        luminance_max_total.append(luminance_max)
        luminance_mean_total.append(luminance_mean)
        #
        # display_text = " - WITH SHADOW MASK - Versions: " + global_config.sm_network_version + "_" + str(global_config.sm_iteration) + \
        #                "<br>" + global_config.ns_network_version + "_" + str(global_config.ns_iteration) + \
        #                "<br> Luminance min: " + str(luminance_min) + "<br> Luminance max: " + str(luminance_max) + \
        #                "<br> Luminance mean: " + str(luminance_mean)

        # visdom_reporter.plot_text(display_text)

    min = np.round(np.mean(luminance_min_total), 4)
    max = np.round(np.mean(luminance_max_total), 4)
    mean = np.round(np.mean(luminance_mean_total), 4)
    display_text = " - WITH SHADOW MASK - Versions: " + global_config.sm_network_version + "_" + str(global_config.sm_iteration) + \
                   "<br>" + global_config.ns_network_version + "_" + str(global_config.ns_iteration) + \
                   "<br> OVERALL Luminance min: " + str(min) + "<br> OVERALL Luminance max: " + str(max) + \
                   "<br> OVERALL Luminance mean: " + str(mean)

    visdom_reporter.plot_text(display_text)

    luminance_min_total = []
    luminance_max_total = []
    luminance_mean_total = []

    for i, (file_name, rgb_ws, _, _, _) in enumerate(shadow_loader, 0):
        rgb_ws = rgb_ws.to(device)

        pbar.update(1)
        luminance_map = tensor_utils.get_luminance(rgb_ws)
        visdom_reporter.plot_image(rgb_ws, "ISTD WS Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
        visdom_reporter.plot_image(luminance_map, "ISTD Luminance Map - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))

        luminance_vector = torch.flatten(luminance_map)
        luminance_min = np.round(torch.min(luminance_vector[luminance_vector > 0.0]).item(), 4)
        luminance_max = np.round(torch.max(luminance_vector[luminance_vector > 0.0]).item(), 4)
        luminance_mean = np.round(torch.mean(luminance_vector[luminance_vector > 0.0]).item(), 4)

        luminance_min_total.append(luminance_min)
        luminance_max_total.append(luminance_max)
        luminance_mean_total.append(luminance_mean)

        visdom_reporter.plot_text(display_text)

    min = np.round(np.mean(luminance_min_total), 4)
    max = np.round(np.mean(luminance_max_total), 4)
    mean = np.round(np.mean(luminance_mean_total), 4)
    display_text = " - NO SHADOW MASK - Versions: " + global_config.sm_network_version + "_" + str(global_config.sm_iteration) + \
                   "<br>" + global_config.ns_network_version + "_" + str(global_config.ns_iteration) + \
                   "<br> OVERALL Luminance min: " + str(min) + "<br> OVERALL Luminance max: " + str(max) + \
                   "<br> OVERALL Luminance mean: " + str(mean)

    visdom_reporter.plot_text(display_text)

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
    yaml_config = yaml_config.format(network_version=opts.shadow_removal_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    global_config.ns_network_version = opts.shadow_removal_version
    global_config.ns_iteration = opts.shadow_removal_iteration
    global_config.ns_network_config = ConfigHolder.getInstance().get_network_config()
    shadow_t = shadow_removal_trainer.ShadowTrainer(device)

    print("---------------------------------------------------------------------------")
    print("Successfully loaded shadow removal network: ", opts.shadow_removal_version, str(opts.shadow_removal_iteration))
    print("Network config: ", global_config.ns_network_config)
    print("---------------------------------------------------------------------------")

    ConfigHolder.destroy()  # for security, destroy config holder since it should no longer be needed

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

    # ConfigHolder.destroy()  # for security, destroy config holder since it should no longer be needed

    analyze_luminance()


if __name__ == "__main__":
    main(sys.argv)