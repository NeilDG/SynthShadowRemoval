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
parser.add_option('--iteration_a', type=int, default="1")
parser.add_option('--iteration_s1', type=int, default="1")
parser.add_option('--iteration_s2', type=int, default="1")
parser.add_option('--net_config_a', type=int)
parser.add_option('--net_config_s1', type=int)
parser.add_option('--net_config_s2', type=int)
parser.add_option('--num_blocks_a', type=int)
parser.add_option('--num_blocks_s1', type=int)
parser.add_option('--num_blocks_s2', type=int)
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_albedo', type=str, help="version_name")
parser.add_option('--version_shading', type=str, help="version_name")
parser.add_option('--version_shadow', type=str, help="version_name")
parser.add_option('--light_angle', type=int, help="Light angle", default = "0")
parser.add_option('--light_color', type=str, help="Light color", default = "225,247,250")

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:16], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

def normalize(light_angle):
    std = light_angle / 360.0
    min = -1.0
    max = 1.0
    scaled = std * (max - min) + min

    return scaled

def prepare_shadow_input(a_tensor, shading_tensor, light_angle):
    light_angle = normalize(light_angle)
    light_angle_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], light_angle), 1)
    concat_input = torch.cat([a_tensor, shading_tensor, light_angle_tensor], 1)
    return concat_input

def prepare_shading_input(a_tensor, albedo_tensor, light_angle):
    light_angle = normalize(light_angle)
    light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], light_angle), 1)
    concat_input = torch.cat([a_tensor, albedo_tensor, light_tensor], 1)
    return concat_input

def produce_rgb(albedo_tensor, shading_tensor, light_color, shadowmap_tensor):
    albedo_tensor = albedo_tensor.transpose(0, 1) * 0.5
    shading_tensor = shading_tensor.transpose(0, 1)
    shadowmap_tensor = shadowmap_tensor.transpose(0, 1)
    light_color = torch.from_numpy(np.asarray(light_color.split(","), dtype = np.int32))

    #normalize/remove normalization
    albedo_tensor = (albedo_tensor * 0.5) + 0.5
    shading_tensor = (shading_tensor * 0.5) + 0.5
    shadowmap_tensor = (shadowmap_tensor * 0.5) + 0.5
    light_color = light_color / 255.0


    rgb_img_like = torch.full_like(albedo_tensor, 0)
    rgb_img_like[0] = torch.clip((albedo_tensor[0] * shading_tensor[0] * light_color[0] * shadowmap_tensor[0]), 0.0, 1.0)
    rgb_img_like[1] = torch.clip((albedo_tensor[1] * shading_tensor[1] * light_color[1] * shadowmap_tensor[1]), 0.0, 1.0)
    rgb_img_like[2] = torch.clip((albedo_tensor[2] * shading_tensor[2] * light_color[2] * shadowmap_tensor[2]), 0.0, 1.0)

    rgb_img_like = rgb_img_like.transpose(0, 1)
    return rgb_img_like

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

    albedo_path = constants.DATASET_ALBEDO_4_PATH
    rgb_path = constants.DATASET_PREFIX_4_PATH + str(opts.light_angle) + "deg/" + "rgb/"
    shading_path =constants.DATASET_PREFIX_4_PATH + str(opts.light_angle) + "deg/" + "shading/"
    shadow_path = constants.DATASET_PREFIX_4_PATH + str(opts.light_angle) + "deg/" + "shadow_map/"

    # Create the dataloader
    shading_loader = dataset_loader.load_shading_test_dataset(rgb_path, shading_path, opts)
    shadow_loader = dataset_loader.load_shadowmap_test_dataset(albedo_path, shadow_path, shading_path, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)

    # Plot some training images
    view_batch, test_a_batch, test_b_batch, test_c_batch = next(iter(shadow_loader))
    test_a_tensor = test_a_batch.to(device)
    test_b_tensor = test_b_batch.to(device)
    test_c_tensor = test_c_batch.to(device)

    show_images(test_a_tensor, "Training - A Images")
    show_images(test_b_tensor, "Training - B Images")
    show_images(test_c_tensor, "Training - C Images")

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

    ALBEDO_CHECKPATH = 'checkpoint/' + opts.version_albedo + "_" + str(opts.iteration_a) + '.pt'
    checkpoint = torch.load(ALBEDO_CHECKPATH, map_location=device)
    G_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_s1 == 1):
        G_shader = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 2):
        G_shader = unet_gan.UnetGenerator(input_nc=7, output_nc=3, num_downs=opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 3):
        G_shader = ffa.FFA(gps=7, blocks=opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 4):
        G_shader = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s1, has_dropout=False).to(device)
    else:
        G_shader = cycle_gan.GeneratorV2(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s1, has_dropout=False, multiply=True).to(device)

    SHADER_CHECKPATH = 'checkpoint/' + opts.version_shading + "_" + str(opts.iteration_s1) + '.pt'
    checkpoint = torch.load(SHADER_CHECKPATH, map_location=device)
    G_shader.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_s2 == 1):
        G_shadow = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 2):
        G_shadow = unet_gan.UnetGenerator(input_nc=7, output_nc=3, num_downs=opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 3):
        G_shadow = ffa.FFA(gps=7, blocks=opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 4):
        G_shadow = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s2, has_dropout=False).to(device)
    else:
        G_shadow = cycle_gan.GeneratorV2(input_nc=7, output_nc=3, n_residual_blocks=opts.num_blocks_s2, has_dropout=False, multiply=True).to(device)

    SHADOW_CHECKPATH = 'checkpoint/' + opts.version_shadow + "_" + str(opts.iteration_s2) + '.pt'
    checkpoint = torch.load(SHADOW_CHECKPATH, map_location=device)
    G_shadow.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])


    print("Loaded checkpt: %s %s %s " % (ALBEDO_CHECKPATH, SHADER_CHECKPATH, SHADOW_CHECKPATH))
    print("===================================================")

    print("Plotting test images...")

    visdom_reporter = plot_utils.VisdomReporter()
    _, rgb_batch, _ = next(iter(shading_loader))
    _, albedo_batch, shadow_batch, shading_batch = next(iter(shadow_loader))
    rgb_tensor = rgb_batch.to(device)
    albedo_tensor = albedo_batch.to(device)
    shading_tensor = shading_batch.to(device)
    shadow_tensor = shadow_batch.to(device)
    rgb2albedo = G_albedo(rgb_tensor)
    rgb2shading = G_shader(prepare_shading_input(rgb_tensor, rgb2albedo, opts.light_angle))
    input2shadow = G_shadow(prepare_shadow_input(rgb_tensor, rgb2shading, opts.light_angle))
    rgb_like = produce_rgb(rgb2albedo, rgb2shading, opts.light_color, input2shadow)

    visdom_reporter.plot_image(rgb_tensor, "Test RGB images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2albedo, "Test RGB 2 Albedo images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2shading, "Test RGB 2 Shading images - " + opts.version_shading + str(opts.iteration_s1))
    # visdom_reporter.plot_image(input2shadow, "Test RGB 2 Shadow images - " + opts.version_shadow + str(opts.iteration_s2))
    visdom_reporter.plot_image(rgb_like, "Test RGB Reconstructed - " + opts.version_albedo + str(opts.iteration_a))

    _, rgb_batch = next(iter(rw_loader))
    rgb_tensor = rgb_batch.to(device)
    rgb2albedo = G_albedo(rgb_tensor)
    rgb2shading = G_shader(prepare_shading_input(rgb_tensor, rgb2albedo, opts.light_angle))
    input2shadow = G_shadow(prepare_shadow_input(rgb_tensor, rgb2shading, opts.light_angle))
    rgb_like = produce_rgb(rgb2albedo, rgb2shading, opts.light_color, input2shadow)

    visdom_reporter.plot_image(rgb_tensor, "RW RGB images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2albedo, "RW RGB 2 Albedo images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2shading, "RW RGB 2 Shading images - " + opts.version_shading + str(opts.iteration_s1))
    # visdom_reporter.plot_image(input2shadow, "RW RGB 2 Shadow images - " + opts.version_shadow + str(opts.iteration_s2))
    visdom_reporter.plot_image(rgb_like, "RW RGB Reconstructed - " + opts.version_albedo + str(opts.iteration_a))




if __name__ == "__main__":
    main(sys.argv)