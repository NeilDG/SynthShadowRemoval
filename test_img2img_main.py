import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from testers import img2img_tester
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="vXX.XX")
parser.add_option('--save_images', type=int, default=0)
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--plot_enabled', type=int, default=1)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_to_load = opts.img_to_load
    global_config.cuda_device = opts.cuda_device
    global_config.style_transfer_version = opts.network_version
    global_config.st_iteration = opts.iteration
    global_config.test_size = 64
    global_config.save_images = opts.save_images

    network_config = ConfigHolder.getInstance().get_network_config()
    dataset_a_version = network_config["dataset_a_version"]
    dataset_b_version = network_config["dataset_b_version"]
    global_config.dataset_target = dataset_a_version

    if (global_config.server_config == 0):  # COARE
        global_config.num_workers = 6
        global_config.disable_progress_bar = True
        global_config.path = "/scratch1/scratch2/neil.delgallego/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using COARE configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 1):  # CCS Cloud
        global_config.num_workers = 12
        global_config.a_path = "/home/jupyter-neil.delgallego/"
        global_config.b_path = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using CCS configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 2):  # RTX 2080Ti
        global_config.num_workers = 6
        global_config.a_path = "C:/Datasets/Places Dataset/*.jpg"
        global_config.b_path = "C:/Datasets/SynthV3_Raw/{dataset_version}/sequence.0/*.camera.png"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using RTX 2080Ti configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 3):  # RTX 3090 PC
        global_config.num_workers = 12
        global_config.a_path = "X:"
        global_config.b_path = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using RTX 3090 configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 4):  # RTX 2070 PC @RL208
        global_config.num_workers = 4
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        global_config.path = "D:/Datasets/SynthV3_Raw/{dataset_version}/sequence.0/"
        print("Using RTX 2070 @RL208 configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 5):  # @TITAN1 - 3
        global_config.num_workers = 4
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        global_config.a_path = "/home/neildelgallego/Places Dataset/*.jpg"
        global_config.b_path = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/sequence.0/*.camera.png"
        print("Using TITAN Workstation configuration. Workers: ", global_config.num_workers)

    if (dataset_a_version == "istd"):
        global_config.a_path = global_config.a_path + "/ISTD_Dataset/train/train_C/*.png"
    elif (dataset_a_version == "srd"):
        global_config.a_path = global_config.a_path + "/SRD_Train/shadow_free/*.jpg"
    else:
        global_config.a_path = global_config.a_path + "/Places Dataset/*.jpg"
    global_config.b_path = global_config.b_path.format(dataset_version=dataset_b_version)


def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/synth2real_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparam_config = ConfigHolder.getInstance().get_hyper_params()
    network_iteration = global_config.st_iteration
    hyperparams_table = hyperparam_config["hyperparams"][network_iteration]
    print("Network iteration:", str(network_iteration), ". Hyper parameters: ", hyperparams_table, " Learning rates: ", network_config["g_lr"], network_config["d_lr"])

    a_path = global_config.a_path
    b_path = global_config.b_path

    print("Dataset path A: ", a_path)
    print("Dataset path B: ", b_path)

    plot_utils.VisdomReporter.initialize()

    test_loader_a, test_count = dataset_loader.load_test_img2img_dataset(a_path, b_path)

    img2img_t = img2img_tester.Img2ImgTester(device)
    start_epoch = global_config.last_epoch_st
    print("---------------------------------------------------------------------------")
    print("Started synth test loop for mode: synth2real", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    steps = global_config.test_size
    needed_progress = int(test_count / steps) + 1
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    with torch.no_grad():
        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_a, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            input_map = {"file_name": file_name, "img_a" : a_batch, "img_b" : b_batch}
            img2img_t.measure_and_store(input_map)
            pbar.update(1)

            if((i + 1) % 4 == 0):
                break

        pbar.close()

        _, a_test_batch, b_test_batch = next(iter(test_loader_a))
        a_test_batch = a_test_batch.to(device)
        b_test_batch = b_test_batch.to(device)
        input_map = {"img_a": a_test_batch, "img_b": b_test_batch}
        if (global_config.plot_enabled == 1):
            img2img_t.visualize_results(input_map, "Synth2Real")
        img2img_t.report_metrics("Synth2Real")

    network_config = ConfigHolder.getInstance().get_network_config()
    dataset_b_version = network_config["dataset_b_version"]

    rgb_noshadows_path = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
    rgb_withshadows_path = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    rgb_noshadows_path = rgb_noshadows_path.format(dataset_version=dataset_b_version)
    rgb_withshadows_path = rgb_withshadows_path.format(dataset_version=dataset_b_version)

    test_rgb_noshadows, test_count = dataset_loader.load_singleimg_dataset(rgb_noshadows_path)
    test_rgb_withshadows, test_count = dataset_loader.load_singleimg_dataset(rgb_withshadows_path)

    #compute total progress
    steps = global_config.test_size
    needed_progress = int(test_count / steps) + 1
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    if (global_config.save_images == 1):
        with torch.no_grad():
            for i, (noshadows_data, withshadows_data) in enumerate(zip(test_rgb_noshadows, test_rgb_withshadows)):
                file_name, img_batch = noshadows_data
                img_batch = img_batch.to(device, non_blocking = True)
                input_map_a = {"file_name": file_name, "img_a": img_batch, "img_b": img_batch}

                file_name, img_batch = withshadows_data
                img_batch = img_batch.to(device, non_blocking = True)
                input_map_b = {"file_name": file_name, "img_a": img_batch, "img_b": img_batch}

                img2img_t.save_images(input_map_a, input_map_b)
                pbar.set_description("Successfully saved images for batch")
                pbar.update(1)

            pbar.close()


if __name__ == "__main__":
    main(sys.argv)