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
parser.add_option('--debug_run', type=int, help="Debug mode?", default=0)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)

def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled
    constants.debug_run = opts.debug_run

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows/"
        constants.albedo_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/unlit/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows/"
        constants.albedo_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/unlit/"
        constants.DATASET_PLACES_PATH = constants.rgb_dir_ws_styled

        print("Using CCS configuration. Workers: ", opts.num_workers)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "/home/neil_delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "/home/neil_delgallego/SynthWeather Dataset 8/albedo/"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "C:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "C:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "C:/Datasets/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 8/train_rgb/*/*.png"
        constants.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 8/train_rgb_noshadows/*/*.png"
        constants.albedo_dir = "C:/Datasets/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "C:/Datasets/SynthWeather Dataset 8/unlit/"
        constants.ws_istd ="C:/Datasets/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "C:/Datasets/ISTD_Dataset/test/test_C/*.png"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 8/train_rgb/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 8/train_rgb_noshadows/"
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
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    plot_utils.VisdomReporter.initialize()

    iid_server_config.IIDServerConfig.initialize(opts.version)
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_network_config_from_version(opts.version)
    print("General config:", general_config)
    print("Network config: ", network_config)

    tf = trainer_factory.TrainerFactory(device, opts)
    tf.initialize_all_trainers(opts)
    iid_op = iid_transforms.IIDTransform()

    # for mode in (["train_albedo_mask", "train_albedo", "train_shading"]):
    # for mode in (["train_shadow", "train_albedo", "train_shading"]):

    #Train shadow
    mode = "train_shadow"
    patch_size = general_config[mode]["patch_size"]
    style_enabled = network_config["style_transferred"]

    if(style_enabled == 1):
        rgb_dir_ws = constants.rgb_dir_ws_styled
        rgb_dir_ns = constants.rgb_dir_ns_styled
    else:
        rgb_dir_ws = constants.rgb_dir_ws
        rgb_dir_ns = constants.rgb_dir_ns

    batch_size = sc_instance.get_batch_size_from_mode(mode, network_config)

    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, patch_size, batch_size, network_config["istd_mix"], opts)
    test_loader_train = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    test_loader_istd = dataset_loader.load_shadow_test_dataset(constants.ws_istd, constants.ns_istd, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode(mode)
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
        for i, (_, rgb_ws_batch, rgb_ns_batch) in enumerate(train_loader, 0):
            rgb_ws_tensor = rgb_ws_batch.to(device)
            rgb_ns_tensor = rgb_ns_batch.to(device)
            rgb_ws_tensor, rgb_ns_tensor, shadow_matte_tensor, _ = iid_op.decompose_shadow(rgb_ws_tensor, rgb_ns_tensor)

            input_map = {"rgb": rgb_ws_tensor, "rgb_ns" : rgb_ns_tensor, "shadow_matte" : shadow_matte_tensor}
            target_map = input_map

            tf.train(mode, epoch, iteration, input_map, target_map)
            iteration = iteration + 1

            if(tf.is_stop_condition_met(mode)):
                break

            if (i % 300 == 0):
                tf.save(mode, epoch, iteration, True)

                if(opts.plot_enabled == 1):
                    tf.visdom_plot(mode, iteration)
                    tf.visdom_visualize(mode, input_map, "Train")

                    _, rgb_ws_batch, rgb_ns_batch = next(itertools.cycle(test_loader_train))
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    rgb_ns_tensor = rgb_ns_batch.to(device)
                    rgb_ws_tensor, rgb_ns_tensor, shadow_matte_tensor, _ = iid_op.decompose_shadow(rgb_ws_tensor, rgb_ns_tensor)

                    input_map = {"rgb": rgb_ws_tensor, "rgb_ns": rgb_ns_tensor, "shadow_matte": shadow_matte_tensor}
                    tf.visdom_visualize(mode, input_map, "Test Synthetic")

                    _, rgb_ws_batch, rgb_ns_batch = next(itertools.cycle(test_loader_istd))
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    rgb_ns_tensor = rgb_ns_batch.to(device)
                    rgb_ws_tensor, rgb_ns_tensor, shadow_matte_tensor, _ = iid_op.decompose_shadow(rgb_ws_tensor, rgb_ns_tensor)

                    input_map = {"rgb": rgb_ws_tensor, "rgb_ns" : rgb_ns_tensor, "shadow_matte" : shadow_matte_tensor}
                    tf.visdom_visualize(mode, input_map, "Test ISTD")

                    _, rgb_ws_batch = next(itertools.cycle(rw_loader))
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    input_map = {"rgb": rgb_ws_tensor}
                    tf.visdom_infer(mode, input_map)



            if (tf.is_stop_condition_met(mode)):
                break


if __name__ == "__main__":
    main(sys.argv)