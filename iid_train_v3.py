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
parser.add_option('--d_lr', type=float, help="LR", default="0.0005")
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
        constants.rgb_dir_ws_v2 = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v2/rgb/*/*.png"
        constants.rgb_dir_ns_v2 = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v2/rgb_noshadows/*/*.png"
        constants.rgb_dir_ws_v3 = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v3/rgb/*/*.png"
        constants.rgb_dir_ns_v3 = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v3/rgb_noshadows/*/*.png"
        constants.ws_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "/scratch1/scratch2/neil.delgallego/ISTD_Dataset/test/test_C/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/rgb/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/rgb_noshadows/*/*.png"
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
        constants.rgb_dir_ws_styled = "C:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "C:/Datasets/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws = "C:/Datasets/SynthWeather Dataset 10/rgb/*/*.png"
        constants.rgb_dir_ns = "C:/Datasets/SynthWeather Dataset 10/rgb_noshadows/*/*.png"
        constants.ws_istd ="C:/Datasets/ISTD_Dataset/test/test_A/*.png"
        constants.ns_istd = "C:/Datasets/ISTD_Dataset/test/test_C/*.png"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/*/*.png"
        constants.rgb_dir_ws_v2 = "E:/SynthWeather Dataset 10/v2/rgb/*/*.png"
        constants.rgb_dir_ns_v2 = "E:/SynthWeather Dataset 10/v2/rgb_noshadows/*/*.png"
        constants.rgb_dir_ws_v3 = "E:/SynthWeather Dataset 10/v3/rgb/*/*.png"
        constants.rgb_dir_ns_v3 = "E:/SynthWeather Dataset 10/v3/rgb_noshadows/*/*.png"
        constants.rgb_dir_ws_v5 = "E:/SynthWeather Dataset 10/v5/rgb/*/*.png"
        constants.rgb_dir_ns_v5 = "E:/SynthWeather Dataset 10/v5/rgb_noshadows/*/*.png"
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
    
    torch.multiprocessing.set_sharing_strategy('file_system')

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    plot_utils.VisdomReporter.initialize()

    constants.network_version = opts.version
    iid_server_config.IIDServerConfig.initialize()
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_network_config_from_version()
    print("General config:", general_config)
    print("Network config: ", network_config)

    tf = trainer_factory.TrainerFactory(device, opts)
    tf.initialize_all_trainers(opts)

    # for mode in (["train_albedo_mask", "train_albedo", "train_shading"]):
    # for mode in (["train_shadow", "train_albedo", "train_shading"]):

    #Train shadow
    mode = "train_shadow"
    patch_size = general_config[mode]["patch_size"]
    dataset_version = network_config["dataset_version"]

    assert dataset_version == "v2" or dataset_version == "v3" or dataset_version == "v5", "Cannot identify dataset version."

    if(dataset_version == "v2"):
        rgb_dir_ws = constants.rgb_dir_ws_v2
        rgb_dir_ns = constants.rgb_dir_ns_v2
    elif(dataset_version == "v3"):
        rgb_dir_ws = constants.rgb_dir_ws_v3
        rgb_dir_ns = constants.rgb_dir_ns_v3
    elif(dataset_version == "v5"):
        rgb_dir_ws = constants.rgb_dir_ws_v5
        rgb_dir_ns = constants.rgb_dir_ns_v5
    else:
        rgb_dir_ws = ""
        rgb_dir_ns = ""

    print("Dataset path: ", rgb_dir_ws, rgb_dir_ns)

    load_size = network_config["load_size_z"]

    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, patch_size, load_size, opts)
    test_loader_train = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    test_loader_istd = dataset_loader.load_shadow_test_dataset(constants.ws_istd, constants.ns_istd, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode(mode)
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
        for i, (_, rgb_ws, rgb_ns, shadow_map) in enumerate(train_loader, 0):
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_map = shadow_map.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns" : rgb_ns, "shadow_map": shadow_map}
            target_map = input_map

            tf.train(mode, epoch, iteration, input_map, target_map)
            iteration = iteration + 1

            if (tf.is_stop_condition_met(mode)):
                break

            if (i % 300 == 0):
                tf.save(mode, epoch, iteration, True)

                if (opts.plot_enabled == 1):
                    tf.visdom_plot(mode, iteration)
                    tf.visdom_visualize(mode, input_map, "Train")

                    _, rgb_ws, rgb_ns = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns}
                    tf.visdom_visualize(mode, input_map, "Test Synthetic")

                    _, rgb_ws, rgb_ns = next(itertools.cycle(test_loader_istd))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns}
                    tf.visdom_visualize(mode, input_map, "Test ISTD")

                    _, rgb_ws_batch = next(itertools.cycle(rw_loader))
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    input_map = {"rgb": rgb_ws_tensor}
                    tf.visdom_infer(mode, input_map)

            if (tf.is_stop_condition_met(mode)):
                break

    #Train shadow refine
    mode = "train_shadow_refine"
    patch_size = general_config[mode]["patch_size"]
    style_enabled = network_config["style_transferred"]
    refine_enabled = network_config["refine_enabled"]
    mix_type = network_config["mix_type"]

    if(refine_enabled == False):
        print("Refinement network training DISABLED. Stopping.")
        return

    print("Refinement network training STARTED...")
    if(style_enabled == 1):
        rgb_dir_ws = constants.rgb_dir_ws_styled
        rgb_dir_ns = constants.rgb_dir_ns_styled
    else:
        rgb_dir_ws = constants.rgb_dir_ws
        rgb_dir_ns = constants.rgb_dir_ns

    batch_size = sc_instance.get_batch_size_from_mode(mode, network_config)

    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, constants.ws_istd, constants.ns_istd, patch_size, batch_size, mix_type, opts)
    test_loader_train = dataset_loader.load_shadow_test_dataset(rgb_dir_ws, rgb_dir_ns, opts)
    test_loader_istd = dataset_loader.load_shadow_test_dataset(constants.ws_istd, constants.ns_istd, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode(mode)
    print("Started Training loop for mode: ", mode, " Set start epoch: ", start_epoch)
    for epoch in range(start_epoch, general_config[mode]["max_epochs"]):
        for i, (_, rgb_ws, rgb_ns, shadow_map) in enumerate(train_loader, 0):
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_map = shadow_map.to(device)

            input_map = {"rgb": rgb_ws, "rgb_ns" : rgb_ns, "shadow_map" : shadow_map}
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

                    _, rgb_ws, rgb_ns = next(itertools.cycle(test_loader_train))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns}
                    shadow_t = tf.get_shadow_trainer()
                    _, rgb2sm = shadow_t.test(input_map)
                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_map": rgb2sm}
                    tf.visdom_visualize(mode, input_map, "Test Synthetic")

                    _, rgb_ws, rgb_ns = next(itertools.cycle(test_loader_istd))
                    rgb_ws = rgb_ws.to(device)
                    rgb_ns = rgb_ns.to(device)

                    input_map = {"rgb": rgb_ws, "rgb_ns" : rgb_ns}
                    _, rgb2sm = shadow_t.test(input_map)
                    input_map = {"rgb": rgb_ws, "rgb_ns": rgb_ns, "shadow_map": rgb2sm}
                    tf.visdom_visualize(mode, input_map, "Test ISTD")

                    _, rgb_ws_batch = next(itertools.cycle(rw_loader))
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    input_map = {"rgb": rgb_ws_tensor}
                    tf.visdom_infer(mode, input_map)

            if (tf.is_stop_condition_met(mode)):
                break


if __name__ == "__main__":
    main(sys.argv)
