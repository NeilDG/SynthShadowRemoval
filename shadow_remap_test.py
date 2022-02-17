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
import kornia

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--iteration', type=int, default="1")
parser.add_option('--net_config', type=int)
parser.add_option('--num_blocks', type=int)
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--desired_light_angle', type=int, default="144")
parser.add_option('--light_color', type=str, help="Light color", default = "255,255,255")
parser.add_option('--mode', type=str, default = "azimuth")

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

def produce_rgb(albedo_tensor, shading_tensor, light_color, shadowmap_tensor):
    albedo_tensor = albedo_tensor.transpose(0, 1)
    shading_tensor = shading_tensor.transpose(0, 1)
    shadowmap_tensor = shadowmap_tensor.transpose(0, 1)
    light_color = torch.from_numpy(np.asarray(light_color.split(","), dtype = np.int32))

    #normalize/remove normalization
    albedo_tensor = (albedo_tensor * 0.5) + 0.5
    shading_tensor = (shading_tensor * 0.5) + 0.5
    shadowmap_tensor = (shadowmap_tensor * 0.5) + 0.5
    light_color = light_color / 255.0

    rgb_img_like = torch.full_like(albedo_tensor, 0)
    rgb_img_like[0] = torch.clip(albedo_tensor[0] * shading_tensor[0] * light_color[0] * shadowmap_tensor, 0.0, 1.0)
    rgb_img_like[1] = torch.clip(albedo_tensor[1] * shading_tensor[1] * light_color[1] * shadowmap_tensor, 0.0, 1.0)
    rgb_img_like[2] = torch.clip(albedo_tensor[2] * shading_tensor[2] * light_color[2] * shadowmap_tensor, 0.0, 1.0)

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

    input_path = constants.DATASET_PREFIX_5_PATH + opts.mode + "/" + "0deg/"
    ground_truth_path = constants.DATASET_PREFIX_5_PATH + opts.mode + "/" +str(opts.desired_light_angle) + "deg/"

    print(input_path, ground_truth_path)

    # Create the dataloader
    shadow_loader = dataset_loader.load_shadowmap_test_recursive_2(ground_truth_path, "albedo", "shadow_map", "shading", True, opts)

    # Plot some training images
    view_batch, test_a_batch, test_b_batch, test_c_batch, test_d_tensor = next(iter(shadow_loader))
    test_a_tensor = test_a_batch.to(device)
    test_b_tensor = test_b_batch.to(device)
    test_c_tensor = test_c_batch.to(device)
    test_d_tensor = test_d_tensor.to(device)

    show_images(test_a_tensor, "Test - A Images")
    show_images(test_b_tensor, "Test - B Images")
    show_images(test_c_tensor, "Test - C Images")
    show_images(test_d_tensor, "Test - D Images")

    visdom_reporter = plot_utils.VisdomReporter()
    # _, _, albedo_batch, _ = next(iter(albedo_loader))
    _, rgb_batch, albedo_batch, shadow_batch, shading_batch = next(iter(shadow_loader))
    rgb_tensor = rgb_batch.to(device)
    albedo_tensor = albedo_batch.to(device)
    shading_tensor = shading_batch.to(device)
    shadow_tensor = shadow_batch.to(device)

    rgb_like = produce_rgb(albedo_tensor, shading_tensor, opts.light_color, shadow_tensor)

    # plot metrics
    rgb_tensor = (rgb_tensor * 0.5) + 0.5

    psnr_rgb = np.round(kornia.losses.psnr(rgb_like, rgb_tensor, max_val=1.0).item(), 4)
    ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, rgb_tensor, 5).item(), 4)
    display_text = "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)
    visdom_reporter.plot_text(display_text)

    visdom_reporter.plot_image(rgb_tensor, "Test RGB images - " + opts.version_name + str(opts.iteration))
    visdom_reporter.plot_image(rgb_like, "Test RGB Reconstructed - " + opts.version_name + str(opts.iteration))




if __name__ == "__main__":
    main(sys.argv)