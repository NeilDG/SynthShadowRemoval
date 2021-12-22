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
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
import constants
from utils import plot_utils

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--iteration_a', type=int, help="Style version?", default="1")
parser.add_option('--iteration_s', type=int, help="Style version?", default="1")
parser.add_option('--net_config_a', type=int)
parser.add_option('--net_config_s', type=int)
parser.add_option('--num_blocks_a', type=int)
parser.add_option('--num_blocks_s', type=int)
parser.add_option('--use_mask', type=int, default = "0")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name_a', type=str, help="version_name")
parser.add_option('--version_name_s', type=str, help="version_name")
parser.add_option('--map_choice', type=str, help="Map choice", default = "albedo")


def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:16], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000)  # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    if(opts.map_choice == "albedo"):
        map_path = constants.DATASET_ALBEDO_DECOMPOSE_PATH
    elif(opts.map_choice == "shading"):
        map_path = constants.DATASET_SHADING_DECOMPOSE_PATH
    else:
        print("Cannot determine map choice. Defaulting to Albedo")
        map_path = constants.DATASET_ALBEDO_PATH

    # Create the dataloader
    test_loader = dataset_loader.load_map_test_dataset(constants.DATASET_RGB_DECOMPOSE_PATH, map_path, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)

    # Plot some training images
    _, a_batch, b_batch, mask_batch = next(iter(test_loader))

    show_images(a_batch, "Training - A Images")
    show_images(b_batch, "Training - B Images")
    show_images(mask_batch, "Training - Mask Images")

    if (opts.net_config_a == 1):
        G_albedo = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 2):
        G_albedo = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 3):
        G_albedo = ffa.FFA(gps=3, blocks=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 4):
        G_albedo = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_a, has_dropout=False).to(device)
    else:
        G_albedo = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_a, has_dropout=False, multiply=True).to(device)

    ALBEDO_CHECKPATH = 'checkpoint/' + opts.version_name_a + "_" + str(opts.iteration_a) + '.pt'
    checkpoint = torch.load(ALBEDO_CHECKPATH, map_location=device)
    G_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_a == 1):
        G_shader = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s).to(device)
    elif (opts.net_config_a == 2):
        G_shader = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=opts.num_blocks_s).to(device)
    elif (opts.net_config_a == 3):
        G_shader = ffa.FFA(gps=3, blocks=opts.num_blocks_s).to(device)
    elif (opts.net_config_a == 4):
        G_shader = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s, has_dropout=False).to(device)
    else:
        G_shader = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s, has_dropout=False, multiply=True).to(device)

    SHADER_CHECKPATH = 'checkpoint/' + opts.version_name_s + "_" + str(opts.iteration_s) + '.pt'
    checkpoint = torch.load(SHADER_CHECKPATH, map_location=device)
    G_shader.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])


    print("Loaded checkpt: %s %s " % (ALBEDO_CHECKPATH, SHADER_CHECKPATH))
    print("===================================================")

    print("Plotting test images...")

    visdom_reporter = plot_utils.VisdomReporter()
    _, rgb_batch, _, _ = next(iter(test_loader))
    rgb_tensor = rgb_batch.to(device)
    rgb2albedo = G_albedo(rgb_tensor) * 0.5 + 0.5
    rgb2shading = G_shader(rgb_tensor) * 0.5 + 0.5
    rgb_like = rgb2albedo * rgb2shading

    visdom_reporter.plot_image(rgb_tensor, "Test RGB images - " + opts.version_name_a + str(opts.iteration_a))
    visdom_reporter.plot_image(rgb2albedo, "Test RGB 2 Albedo images - " + opts.version_name_s + str(opts.iteration_s))
    visdom_reporter.plot_image(rgb2shading, "Test RGB 2 Shading images - " + opts.version_name_s + str(opts.iteration_s))
    visdom_reporter.plot_image(rgb_like, "Test RGB Reconstructed - " + opts.version_name_a + str(opts.iteration_a))

    _, rgb_batch = next(iter(rw_loader))
    rgb_tensor = rgb_batch.to(device)
    rgb2albedo = G_albedo(rgb_tensor) * 0.5 + 0.5
    rgb2shading = G_shader(rgb_tensor) * 0.5 + 0.5
    rgb_like = rgb2albedo * rgb2shading

    visdom_reporter.plot_image(rgb_tensor, "RW RGB images - " + opts.version_name_a + str(opts.iteration_a))
    visdom_reporter.plot_image(rgb2albedo, "RW RGB 2 Albedo images - " + opts.version_name_s + str(opts.iteration_s))
    visdom_reporter.plot_image(rgb2shading, "RW RGB 2 Shading images - " + opts.version_name_s + str(opts.iteration_s))
    visdom_reporter.plot_image(rgb_like, "RW RGB Reconstructed - " + opts.version_name_a + str(opts.iteration_a))




if __name__ == "__main__":
    main(sys.argv)