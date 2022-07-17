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
from trainers import iid_trainer
from trainers import early_stopper
from transforms import iid_transforms
import constants
from utils import plot_utils
from trainers import trainer_factory

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--version', type=str, default="")
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)

def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/unlit/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/unlit/"
        constants.DATASET_PLACES_PATH = constants.rgb_dir_ws

        print("Using CCS configuration. Workers: ", opts.num_workers)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/home/neil_delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "/home/neil_delgallego/SynthWeather Dataset 8/albedo/"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "C:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "C:/Datasets/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "C:/Datasets/SynthWeather Dataset 8/unlit/"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
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

    print(constants.rgb_dir_ws, constants.albedo_dir)
    plot_utils.VisdomReporter.initialize()

    start_epoch = 0
    iteration = 0

    iid_server_config.IIDServerConfig.initialize(opts.version)
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_network_config_from_version(opts.version)
    print("General config:", general_config)
    print("Network config: ", network_config)

    tf = trainer_factory.TrainerFactory(device, opts)
    iid_op = iid_transforms.IIDTransform()

    for mode in (["train_albedo_mask", "train_albedo", "train_shading"]):
    # for mode in (["train_albedo", "train_shading"]):
        patch_size = general_config[mode]["patch_size"]
        batch_size = sc_instance.get_batch_size_from_mode(mode, network_config)
        train_loader = dataset_loader.load_iid_datasetv2_train(constants.rgb_dir_ws, constants.rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, patch_size, batch_size, opts)
        test_loader = dataset_loader.load_iid_datasetv2_test(constants.rgb_dir_ws, constants.rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, 256, opts)
        rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

        print("Started Training loop for mode: ", mode)
        iteration = 0

        for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
            for i, (train_data, test_data, rw_data) in enumerate(zip(train_loader, test_loader, itertools.cycle(rw_loader))):
                _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = train_data
                rgb_ws_tensor = rgb_ws_batch.to(device)
                rgb_ns_tensor = rgb_ns_batch.to(device)
                albedo_tensor = albedo_batch.to(device)
                unlit_tensor = unlit_batch.to(device)
                rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

                input_map = {"rgb": rgb_ws_tensor, "albedo": albedo_tensor, "unlit": unlit_tensor, "shading" : shading_tensor, "shadow" : shadow_tensor}
                target_map = input_map

                tf.train(mode, epoch, iteration, input_map, target_map)
                iteration = iteration + 1

                if(tf.is_stop_condition_met(mode)):
                    break

                if (i % 300 == 0):
                    tf.visdom_plot(mode, iteration)
                    tf.visdom_visualize(mode, input_map, "Train")

                    _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = test_data
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    rgb_ns_tensor = rgb_ns_batch.to(device)
                    albedo_tensor = albedo_batch.to(device)
                    unlit_tensor = unlit_batch.to(device)

                    rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)
                    input_map = {"rgb": rgb_ws_tensor, "albedo": albedo_tensor, "unlit": unlit_tensor, "shading" : shading_tensor, "shadow" : shadow_tensor}
                    tf.visdom_visualize(mode, input_map, "Test")

                    # _, rgb_ws_batch = rw_data
                    # rgb_ws_tensor = rgb_ws_batch.to(device)
                    # input_map = {"rgb": rgb_ws_tensor}
                    # tf.visdom_infer(mode, input_map)

                    tf.save(mode, epoch, iteration, True)

            if (tf.is_stop_condition_met(mode)):
                break


if __name__ == "__main__":
    main(sys.argv)