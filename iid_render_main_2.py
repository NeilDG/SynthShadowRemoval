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
from loaders import image_dataset

parser = OptionParser()
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--iteration_a', type=int, default="1")
parser.add_option('--iteration_s1', type=int, default="1")
parser.add_option('--iteration_s2', type=int, default="1")
parser.add_option('--iteration_s3', type=int, default="1")
parser.add_option('--net_config_a', type=int)
parser.add_option('--net_config_s1', type=int)
parser.add_option('--net_config_s2', type=int)
parser.add_option('--net_config_s3', type=int)
parser.add_option('--num_blocks_a', type=int)
parser.add_option('--num_blocks_s1', type=int)
parser.add_option('--num_blocks_s2', type=int)
parser.add_option('--num_blocks_s3', type=int)
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_albedo', type=str, help="version_name")
parser.add_option('--version_shading', type=str, help="version_name")
parser.add_option('--version_shadow', type=str, help="version_name")
parser.add_option('--version_shadow_remap', type=str, help="version_name")
parser.add_option('--mode', type=str, default = "elevation")
parser.add_option('--light_color', type=str, help="Light color", default = "225,247,250")
parser.add_option('--test_code', type=str, default = "111") #Enable albedo - shading - shadow remapping?

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
    # light_angle = normalize(light_angle)
    # light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], light_angle), 1)
    # concat_input = torch.cat([a_tensor, albedo_tensor, light_tensor], 1)
    # return concat_input
    return a_tensor

def produce_rgb(albedo_tensor, shading_tensor, light_color, shadowmap_tensor):
    albedo_tensor = albedo_tensor.transpose(0, 1)
    shading_tensor = shading_tensor.transpose(0, 1)
    shadowmap_tensor = shadowmap_tensor.transpose(0, 1)
    light_color = torch.from_numpy(np.asarray(light_color.split(","), dtype = np.int32))

    print("Shading Range: ", torch.min(shading_tensor).item(), torch.max(shading_tensor).item(), " Mean: ", torch.mean(shading_tensor).item())
    print("ShadowMap Range: ", torch.min(shadowmap_tensor).item(), torch.max(shadowmap_tensor).item(), " Mean: ", torch.mean(shading_tensor).item())
    print("Light Range: ", light_color)

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

    albedo_dir = constants.DATASET_ALBEDO_6_PATH
    shading_dir = constants.DATASET_PREFIX_6_PATH + "shading/"
    rgb_dir = constants.DATASET_PREFIX_6_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "rgb/"
    shadow_dir = constants.DATASET_PREFIX_6_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "shadow_map/"

    print(rgb_dir, albedo_dir, shading_dir, shadow_dir)

    # Create the dataloader
    input_loader = dataset_loader.load_map_test_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)

    # Plot some training images
    # view_batch, test_a_batch, test_b_batch, test_c_batch, test_d_batch, _ = next(iter(input_loader))
    # test_a_tensor = test_a_batch.to(device)
    # test_b_tensor = test_b_batch.to(device)
    # test_c_tensor = test_c_batch.to(device)
    # test_d_tensor = test_d_batch.to(device)
    #
    # show_images(test_a_tensor, "Input - A Images")
    # show_images(test_b_tensor, "Input - B Images")
    # show_images(test_c_tensor, "Input - C Images")
    # show_images(test_d_tensor, "Input - D Images")
    #
    # view_batch, test_a_batch, test_b_batch, test_c_batch, test_d_batch, _ = next(iter(target_loader))
    # test_a_tensor = test_a_batch.to(device)
    # test_b_tensor = test_b_batch.to(device)
    # test_c_tensor = test_c_batch.to(device)
    # test_d_tensor = test_d_batch.to(device)
    #
    # show_images(test_a_tensor, "Target - A Images")
    # show_images(test_b_tensor, "Target - B Images")
    # show_images(test_c_tensor, "Target - C Images")
    # show_images(test_d_tensor, "Target - D Images")

    if (opts.net_config_a == 1):
        G_albedo = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 2):
        G_albedo = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 3):
        G_albedo = ffa.FFA(gps=3, blocks=opts.num_blocks_a).to(device)
    elif (opts.net_config_a == 4):
        G_albedo = cycle_gan.GeneratorV3(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_a).to(device)
    else:
        G_albedo = unet_gan.UnetGeneratorV2(input_nc=3, output_nc=3, num_downs=opts.num_blocks_a).to(device)

    ALBEDO_CHECKPATH = 'checkpoint/' + opts.version_albedo + "_" + str(opts.iteration_a) + '.pt'
    checkpoint = torch.load(ALBEDO_CHECKPATH, map_location=device)
    G_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_s1 == 1):
        G_shader = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 2):
        G_shader = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 3):
        G_shader = ffa.FFAWithBackbone(input_nc=3, blocks = opts.num_blocks_s1).to(device)
    elif (opts.net_config_s1 == 4):
        G_shader = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s1, has_dropout=False).to(device)
    else:
        G_shader = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=opts.num_blocks_s1, has_dropout=False, multiply=True).to(device)

    SHADER_CHECKPATH = 'checkpoint/' + opts.version_shading + "_" + str(opts.iteration_s1) + '.pt'
    checkpoint = torch.load(SHADER_CHECKPATH, map_location=device)
    G_shader.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_s2 == 1):
        G_shadow = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 2):
        G_shadow = unet_gan.UnetGenerator(input_nc=3, output_nc=1, num_downs=opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 3):
        G_shadow = ffa.FFAWithBackbone(input_nc=3, blocks = opts.num_blocks_s2).to(device)
    elif (opts.net_config_s2 == 4):
        G_shadow = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=opts.num_blocks_s2, has_dropout=False).to(device)
    else:
        G_shadow = cycle_gan.GeneratorV2(input_nc=3, output_nc=1, n_residual_blocks=opts.num_blocks_s2, has_dropout=False, multiply=True).to(device)

    SHADOW_CHECKPATH = 'checkpoint/' + opts.version_shadow + "_" + str(opts.iteration_s2) + '.pt'
    checkpoint = torch.load(SHADOW_CHECKPATH, map_location=device)
    G_shadow.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    if (opts.net_config_s3 == 1):
        G_shadow_remap = cycle_gan.Generator(input_nc=5, output_nc=1, n_residual_blocks=opts.num_blocks_s3).to(device)
    elif (opts.net_config_s3 == 2):
        G_shadow_remap = unet_gan.UnetGenerator(input_nc=5, output_nc=1, num_downs=opts.num_blocks_s3).to(device)
    else:
        G_shadow_remap = cycle_gan.Generator(input_nc=5, output_nc=1, n_residual_blocks=opts.num_blocks_s3, has_dropout=False).to(device)

    SHADOW_REMAP_CHECKPATH = 'checkpoint/' + opts.version_shadow_remap + "_" + str(opts.iteration_s3) + '.pt'
    checkpoint = torch.load(SHADOW_REMAP_CHECKPATH, map_location=device)
    G_shadow_remap.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])

    print("Loaded checkpt: %s %s %s %s " % (ALBEDO_CHECKPATH, SHADER_CHECKPATH, SHADOW_CHECKPATH, SHADOW_REMAP_CHECKPATH))
    print("===================================================")

    print("Plotting test images...")
    visdom_reporter = plot_utils.VisdomReporter()
    _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(input_loader))
    input_rgb_tensor = input_rgb_batch.to(device)
    target_rgb_tensor = target_rgb_batch.to(device)
    albedo_tensor = albedo_batch.to(device)
    shading_tensor = shading_batch.to(device)
    target_shadow_tensor = target_shadow_batch.to(device)
    light_angle_tensor = light_angle_batch.to(device)

    if(opts.test_code[0] == "1"):
        rgb2albedo = G_albedo(target_rgb_tensor)
    else:
        rgb2albedo = albedo_tensor

    if(opts.test_code[1] == "1"):
        rgb2shading = G_shader(target_rgb_tensor)
    else:
        rgb2shading = shading_tensor

    if (opts.test_code[2] == "1"):
        input2shadow = G_shadow(input_rgb_tensor)
        concat_input = torch.cat([input2shadow, input_rgb_tensor, light_angle_tensor], 1)
        input2shadow = G_shadow_remap(concat_input)
    else:
        input2shadow = target_shadow_tensor

    rgb_like = produce_rgb(rgb2albedo, rgb2shading, opts.light_color, input2shadow)

    #plot metrics
    rgb2albedo = (rgb2albedo * 0.5) + 0.5
    albedo_tensor = (albedo_tensor * 0.5) + 0.5
    rgb2shading = (rgb2shading * 0.5) + 0.5
    shading_tensor = (shading_tensor * 0.5) + 0.5
    input2shadow = (input2shadow * 0.5) + 0.5
    target_shadow_tensor = (target_shadow_tensor * 0.5) + 0.5
    target_rgb_tensor = (target_rgb_tensor * 0.5) + 0.5

    psnr_albedo = np.round(kornia.losses.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
    psnr_shading = np.round(kornia.losses.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
    ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
    psnr_shadow = np.round(kornia.losses.psnr(input2shadow, target_shadow_tensor, max_val=1.0).item(), 4)
    ssim_shadow = np.round(1.0 - kornia.losses.ssim_loss(input2shadow, target_shadow_tensor, 5).item(), 4)
    psnr_rgb = np.round(kornia.losses.psnr(rgb_like, target_rgb_tensor, max_val=1.0).item(), 4)
    ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, target_rgb_tensor, 5).item(), 4)
    display_text = "Versions: " + opts.version_albedo + str(opts.iteration_a) + "<br>" \
                   + opts.version_shading + str(opts.iteration_s1) + "<br>" \
                   + opts.version_shadow + str(opts.iteration_s2) + "<br>" \
                   + opts.version_shadow_remap + str(opts.iteration_s3) + "<br>" \
                          "<br> Albedo PSNR: " + str(psnr_albedo) + \
                          "<br> Albedo SSIM: " + str(ssim_albedo) + "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                            "<br> Shadow PSNR: " + str(psnr_shadow) + "<br> Shadow SSIM: " +str(ssim_shadow) + \
                            "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " +str(ssim_rgb)
    visdom_reporter.plot_text(display_text)


    #remove artifacts
    # albedo_tensor = torch.clip(albedo_tensor, 0.1, 1.0)
    # shading_tensor = torch.clip(shading_tensor, 0.1, 1.0)
    # shadow_tensor = torch.clip(shadow_tensor, 0.1, 1.0)
    # rgb2albedo = torch.clip(rgb2albedo, 0.1, 1.0)
    # rgb2shading = torch.clip(rgb2shading, 0.1, 1.0)
    # input2shadow = torch.clip(input2shadow, 0.1, 1.0)

    if (opts.test_code[0] == "1"):
        visdom_reporter.plot_image(albedo_tensor, "Test Albedo images - " + opts.version_albedo + str(opts.iteration_a))
        visdom_reporter.plot_image(rgb2albedo, "Test RGB 2 Albedo images - " + opts.version_albedo + str(opts.iteration_a))

    if (opts.test_code[1] == "1"):
        visdom_reporter.plot_image(shading_tensor, "Test Shading images - " + opts.version_shading + str(opts.iteration_s1))
        visdom_reporter.plot_image(rgb2shading, "Test RGB 2 Shading images - " + opts.version_shading + str(opts.iteration_s1))

    if (opts.test_code[2] == "1"):
        visdom_reporter.plot_image(target_shadow_tensor, "Target Shadow images - " + opts.version_shadow + str(opts.iteration_s2))
        visdom_reporter.plot_image(input2shadow, "Test RGB 2 Shadow images - " + opts.version_shadow + str(opts.iteration_s2))

    visdom_reporter.plot_image(input_rgb_tensor, "A: RGB images - " + opts.version_shadow_remap + str(opts.iteration_s3))
    visdom_reporter.plot_image(target_rgb_tensor, "B: RGB images - " + opts.version_shadow_remap + str(opts.iteration_s3))
    visdom_reporter.plot_image(rgb_like, "B: RGB Reconstructed - " + opts.version_shadow_remap + str(opts.iteration_s3))

    _, rgb_batch = next(iter(rw_loader))
    target_rgb_tensor = rgb_batch.to(device)
    rgb2albedo = G_albedo(target_rgb_tensor)
    rgb2shading = G_shader(target_rgb_tensor)
    input2shadow = G_shadow(target_rgb_tensor)
    #desired light angle
    light_angle = image_dataset.normalize(36)
    light_angle_tensor = torch.full_like(input2shadow, light_angle)
    concat_input = torch.cat([input2shadow, target_rgb_tensor, light_angle_tensor], 1)
    input2shadow = G_shadow_remap(concat_input)

    rgb_like = produce_rgb(rgb2albedo, rgb2shading, opts.light_color, input2shadow)

    visdom_reporter.plot_image(target_rgb_tensor, "RW RGB images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2albedo, "RW RGB 2 Albedo images - " + opts.version_albedo + str(opts.iteration_a))
    # visdom_reporter.plot_image(rgb2shading, "RW RGB 2 Shading images - " + opts.version_shading + str(opts.iteration_s1))
    # visdom_reporter.plot_image(input2shadow, "RW RGB 2 Shadow images - " + opts.version_shadow + str(opts.iteration_s2))
    visdom_reporter.plot_image(rgb_like, "RW RGB Reconstructed - " + opts.version_albedo + str(opts.iteration_a))




if __name__ == "__main__":
    main(sys.argv)