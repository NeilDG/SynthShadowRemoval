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
from loaders import dataset_loader
from trainers import relighting_trainer
from trainers import early_stopper
from utils import tensor_utils
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--rgb_l1_weight', type=float, help="Weight", default="1.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--mode', type=str, default="azimuth")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--debug_mode', type=int, default=0)

# --img_to_load=-1 --load_previous=1
# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.RELIGHTING_VERSION = opts.version_name
    constants.RELIGHTING_CHECKPATH = 'checkpoint/' + constants.RELIGHTING_VERSION + "_" + constants.ITERATION + '.pt'
    constants.plot_enabled = opts.plot_enabled

    if (opts.debug_mode == 1):
        constants.early_stop_threshold = 0
        constants.min_epochs = 1

    # COARE
    if (constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_PREFIX_6_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 7/"
        constants.DATASET_ALBEDO_6_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 7/albedo/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        opts.num_workers = 12
        constants.DATASET_PREFIX_6_PATH = "/home/jupyter-neil.delgallego/SynthWeather Dataset 7/"
        constants.DATASET_ALBEDO_6_PATH = "/home/jupyter-neil.delgallego/SynthWeather Dataset 7/albedo/"
        # constants.DATASET_PLACES_PATH = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        constants.DATASET_PLACES_PATH = constants.DATASET_PREFIX_6_PATH

        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)

    # GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_PREFIX_6_PATH = "/home/neil_delgallego/SynthWeather Dataset 6/"
        constants.DATASET_ALBEDO_6_PATH = "/home/neil_delgallego/SynthWeather Dataset 6/albedo/"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "D:/Datasets/Places Dataset/*.jpg"
        constants.DATASET_PREFIX_6_PATH = "D:/Datasets/SynthWeather Dataset 6/"
        constants.DATASET_ALBEDO_6_PATH = "D:/Datasets/SynthWeather Dataset 6/albedo/"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers, " ", opts.version_name)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.DATASET_PREFIX_6_PATH = "E:/SynthWeather Dataset 7/"
        constants.DATASET_ALBEDO_6_PATH = "E:/SynthWeather Dataset 7/albedo/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers, " ", opts.version_name)


def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:16], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    # manualSeed = random.randint(1, 10000)  # use if you want new results
    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    rgb_dir = "E:/SynthWeather Dataset 8/train_rgb/"
    albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
    scene_root = rgb_dir

    train_loader = dataset_loader.load_relighting_train_dataset(rgb_dir, albedo_dir, scene_root, opts)
    test_loader = dataset_loader.load_relighting_test_dataset(rgb_dir, albedo_dir, scene_root, opts)

    index = 0
    start_epoch = 0
    iteration = 0

    trainer = relighting_trainer.RelightingTrainer(device, opts)
    trainer.update_penalties()

    if (opts.test_mode == 1):
        print("Plotting test images...")
        _, input_rgb_batch, albedo_batch, _ = next(iter(train_loader))

        trainer.visdom_visualize(input_rgb_batch, albedo_batch)

    else:
        print("Starting Training Loop...")
        last_metric = 10000.0
        stopper_method = early_stopper.EarlyStopper(constants.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, last_metric)
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, input_rgb_batch, albedo_batch, scene_idx_batch = train_data
                rgb_tensor = input_rgb_batch.to(device)
                albedo_tensor = albedo_batch.to(device)
                scene_idx_tensor = scene_idx_batch.to(device)

                trainer.visdom_visualize(rgb_tensor, albedo_tensor)
                print(scene_idx_tensor)



if __name__ == "__main__":
    main(sys.argv)