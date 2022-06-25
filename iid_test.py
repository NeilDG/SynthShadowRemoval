import glob
import random
import sys
from optparse import OptionParser
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np

from loaders import dataset_loader
from trainers import iid_trainer
from transforms import iid_transforms
from utils import tensor_utils
from utils import plot_utils
import constants
import cv2
import kornia

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--rgb_l1_weight', type=float, help="Weight", default="1.0")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--da_enabled', type=int, default=0)
parser.add_option('--da_version_name', type=str, default="")
parser.add_option('--albedo_mode', type=int, default="0")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--mode', type=str, default="azimuth")
parser.add_option('--input_path', type=str)
parser.add_option('--output_path', type=str)
parser.add_option('--img_size', type=int, default=(256, 256))

def measure_performance():
    visdom_reporter = plot_utils.VisdomReporter()

    GTA_BASE_PATH = "E:/IID-TestDataset/GTA/"
    RGB_PATH = GTA_BASE_PATH + "/input/"
    ALBEDO_PATH = GTA_BASE_PATH + "/albedo/"

    RESULT_A_PATH = GTA_BASE_PATH + "/li_eccv18/"
    RESULT_B_PATH = GTA_BASE_PATH + "/yu_cvpr19/"
    RESULT_C_PATH = GTA_BASE_PATH + "/yu_eccv20/"
    RESULT_D_PATH = GTA_BASE_PATH + "/zhu_iccp21/"
    RESULT_E_PATH = GTA_BASE_PATH + "/ours/"

    rgb_list = glob.glob(RGB_PATH + "*.png")
    albedo_list = glob.glob(ALBEDO_PATH + "*.png")
    a_list = glob.glob(RESULT_A_PATH + "*.png")
    b_list = glob.glob(RESULT_B_PATH + "*.png")
    c_list = glob.glob(RESULT_C_PATH + "*.png")
    d_list = glob.glob(RESULT_D_PATH + "*.png")
    e_list = glob.glob(RESULT_E_PATH + "*.png")

    IMG_SIZE = (320, 240)

    # albedo_tensor = tensor_utils.load_metric_compatible_img(albedo_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    # a_tensor = tensor_utils.load_metric_compatible_img(a_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    # b_tensor = tensor_utils.load_metric_compatible_img(b_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    # c_tensor = tensor_utils.load_metric_compatible_img(c_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    # d_tensor = tensor_utils.load_metric_compatible_img(d_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)

    albedo_tensor = None
    albedo_a_tensor = None
    albedo_b_tensor = None
    albedo_c_tensor = None
    albedo_d_tensor = None
    albedo_e_tensor = None

    shading_tensor = None
    shading_a_tensor = None
    shading_b_tensor = None
    shading_c_tensor = None
    shading_d_tensor = None

    rgb_tensor = None
    rgb_a_tensor = None
    rgb_b_tensor = None
    rgb_c_tensor = None
    rgb_d_tensor = None


    for i, (rgb_path, albedo_path, a_path, b_path, c_path, d_path, e_path) in enumerate(zip(rgb_list, albedo_list, a_list, b_list, c_list, d_list, e_list)):
        albedo_img = tensor_utils.load_metric_compatible_albedo(albedo_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_a_img = tensor_utils.load_metric_compatible_albedo(a_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_b_img = tensor_utils.load_metric_compatible_albedo(b_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_c_img = tensor_utils.load_metric_compatible_albedo(c_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_d_img = tensor_utils.load_metric_compatible_albedo(d_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        albedo_e_img = tensor_utils.load_metric_compatible_albedo(e_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)

        psnr_albedo_a = np.round(kornia.metrics.psnr(albedo_a_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(albedo_a_img, albedo_img, 5).item(), 4)
        psnr_albedo_b = np.round(kornia.metrics.psnr(albedo_b_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(albedo_b_img, albedo_img, 5).item(), 4)
        psnr_albedo_c = np.round(kornia.metrics.psnr(albedo_c_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(albedo_c_img, albedo_img, 5).item(), 4)
        psnr_albedo_d = np.round(kornia.metrics.psnr(albedo_d_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(albedo_d_img, albedo_img, 5).item(), 4)
        psnr_albedo_e = np.round(kornia.metrics.psnr(albedo_e_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_e = np.round(1.0 - kornia.losses.ssim_loss(albedo_e_img, albedo_img, 5).item(), 4)
        display_text = "Image " +str(i)+ " Albedo <br>" \
                                     "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                                     "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                                     "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                                     "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>" \
                                     "Ours PSNR: " + str(psnr_albedo_e) + "<br> SSIM: " + str(ssim_albedo_e) + "<br>"

        visdom_reporter.plot_text(display_text)

        if(i == 0):
            albedo_tensor = albedo_img
            albedo_a_tensor = albedo_a_img
            albedo_b_tensor = albedo_b_img
            albedo_c_tensor = albedo_c_img
            albedo_d_tensor = albedo_d_img
            albedo_e_tensor = albedo_e_img
        else:
            albedo_tensor = torch.cat([albedo_tensor, albedo_img], 0)
            albedo_a_tensor = torch.cat([albedo_a_tensor, albedo_a_img], 0)
            albedo_b_tensor = torch.cat([albedo_b_tensor, albedo_b_img], 0)
            albedo_c_tensor = torch.cat([albedo_c_tensor, albedo_c_img], 0)
            albedo_d_tensor = torch.cat([albedo_d_tensor, albedo_d_img], 0)
            albedo_e_tensor = torch.cat([albedo_e_tensor, albedo_e_img], 0)

        # compute shading
        # rgb_img = tensor_utils.load_metric_compatible_img(rgb_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        # albedo_img = tensor_utils.load_metric_compatible_albedo(albedo_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        #
        # a_img = tensor_utils.load_metric_compatible_img(a_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        # b_img = tensor_utils.load_metric_compatible_img(b_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        # c_img = tensor_utils.load_metric_compatible_img(c_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        # d_img = tensor_utils.load_metric_compatible_img(d_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        #
        # shading_img = derive_shading(rgb_img, albedo_img)
        # shading_a_img = derive_shading(rgb_img, a_img)
        # shading_b_img = derive_shading(rgb_img, b_img)
        # shading_c_img = derive_shading(rgb_img, c_img)
        # shading_d_img = derive_shading(rgb_img, d_img)
        #
        # psnr_shading_a = np.round(kornia.metrics.psnr(shading_a_img, shading_img, max_val=1.0).item(), 4)
        # ssim_shading_a = np.round(1.0 - kornia.losses.ssim_loss(shading_a_img, shading_img, 5).item(), 4)
        # psnr_shading_b = np.round(kornia.metrics.psnr(shading_b_img, shading_img, max_val=1.0).item(), 4)
        # ssim_shading_b = np.round(1.0 - kornia.losses.ssim_loss(shading_b_img, shading_img, 5).item(), 4)
        # psnr_shading_c = np.round(kornia.metrics.psnr(shading_c_img, shading_img, max_val=1.0).item(), 4)
        # ssim_shading_c = np.round(1.0 - kornia.losses.ssim_loss(shading_c_img, shading_img, 5).item(), 4)
        # psnr_shading_d = np.round(kornia.metrics.psnr(shading_d_img, shading_img, max_val=1.0).item(), 4)
        # ssim_shading_d = np.round(1.0 - kornia.losses.ssim_loss(shading_d_img, shading_img, 5).item(), 4)
        # display_text = "Image " +str(i)+ " Shading <br>" \
        #                              "li_eccv18 PSNR: " + str(psnr_shading_a) + "<br> SSIM: " + str(ssim_shading_a) + "<br>" \
        #                              "yu_cvpr19 PSNR: " + str(psnr_shading_b) + "<br> SSIM: " + str(ssim_shading_b) + "<br>" \
        #                              "yu_eccv20 PSNR: " + str(psnr_shading_c) + "<br> SSIM: " + str(ssim_shading_c) + "<br>" \
        #                              "zhu_iccp21 PSNR: " + str(psnr_shading_d) + "<br> SSIM: " + str(ssim_shading_d) + "<br>"
        #
        # visdom_reporter.plot_text(display_text)
        #
        # if (i == 0):
        #     shading_tensor = shading_img
        #     shading_a_tensor = shading_a_img
        #     shading_b_tensor = shading_b_img
        #     shading_c_tensor = shading_c_img
        #     shading_d_tensor = shading_d_img
        # else:
        #     shading_tensor = torch.cat([shading_tensor, shading_img], 0)
        #     shading_a_tensor = torch.cat([shading_a_tensor, shading_a_img], 0)
        #     shading_b_tensor = torch.cat([shading_b_tensor, shading_b_img], 0)
        #     shading_c_tensor = torch.cat([shading_c_tensor, shading_c_img], 0)
        #     shading_d_tensor = torch.cat([shading_d_tensor, shading_d_img], 0)

        #rgb reconstruction
        # rgb_gt = tensor_utils.load_metric_compatible_img(rgb_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        # rgb_a = reconstruct_rgb(albedo_a_img, shading_a_img)
        # rgb_b = reconstruct_rgb(albedo_b_img, shading_b_img)
        # rgb_c = reconstruct_rgb(albedo_c_img, shading_c_img)
        # albedo_d_img = rgb_gt / shading_d_img
        # rgb_d = reconstruct_rgb(albedo_d_img, shading_d_img)
        #
        # print(np.shape(rgb_gt), np.shape(rgb_a))

        # psnr_rgb_a = np.round(kornia.metrics.psnr(rgb_a, rgb_gt, max_val=1.0).item(), 4)
        # ssim_rgb_a = np.round(1.0 - kornia.losses.ssim_loss(rgb_a, rgb_gt, 5).item(), 4)
        # psnr_rgb_b = np.round(kornia.metrics.psnr(rgb_b, rgb_gt, max_val=1.0).item(), 4)
        # ssim_rgb_b = np.round(1.0 - kornia.losses.ssim_loss(rgb_b, rgb_gt, 5).item(), 4)
        # psnr_rgb_c = np.round(kornia.metrics.psnr(rgb_c, rgb_gt, max_val=1.0).item(), 4)
        # ssim_rgb_c = np.round(1.0 - kornia.losses.ssim_loss(rgb_c, rgb_gt, 5).item(), 4)
        # psnr_rgb_d = np.round(kornia.metrics.psnr(rgb_d, rgb_gt, max_val=1.0).item(), 4)
        # ssim_rgb_d = np.round(1.0 - kornia.losses.ssim_loss(rgb_d, rgb_gt, 5).item(), 4)
        # display_text = "Image " +str(i)+ " RGB <br>" \
        #                              "li_eccv18 PSNR: " + str(psnr_rgb_a) + "<br> SSIM: " + str(ssim_rgb_a) + "<br>" \
        #                              "yu_cvpr19 PSNR: " + str(psnr_rgb_b) + "<br> SSIM: " + str(ssim_rgb_b) + "<br>" \
        #                              "yu_eccv20 PSNR: " + str(psnr_rgb_c) + "<br> SSIM: " + str(ssim_rgb_c) + "<br>" \
        #                              "zhu_iccp21 PSNR: " + str(psnr_rgb_d) + "<br> SSIM: " + str(ssim_rgb_d) + "<br>"
        #
        # visdom_reporter.plot_text(display_text)

        # if (i == 0):
        #     rgb_tensor = rgb_gt
        #     rgb_a_tensor = rgb_a
        #     rgb_b_tensor = rgb_b
        #     rgb_c_tensor = rgb_c
        #     rgb_d_tensor = rgb_d
        # else:
        #     rgb_tensor = torch.cat([rgb_tensor, rgb_gt], 0)
        #     rgb_a_tensor = torch.cat([rgb_a_tensor, rgb_a], 0)
        #     rgb_b_tensor = torch.cat([rgb_b_tensor, rgb_b], 0)
        #     rgb_c_tensor = torch.cat([rgb_c_tensor, rgb_c], 0)
        #     rgb_d_tensor = torch.cat([rgb_d_tensor, rgb_d], 0)

    psnr_albedo_a = np.round(kornia.metrics.psnr(albedo_a_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(albedo_a_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_b = np.round(kornia.metrics.psnr(albedo_b_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(albedo_b_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_c = np.round(kornia.metrics.psnr(albedo_c_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(albedo_c_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_d = np.round(kornia.metrics.psnr(albedo_d_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(albedo_d_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_e = np.round(kornia.metrics.psnr(albedo_e_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_e = np.round(1.0 - kornia.losses.ssim_loss(albedo_e_tensor, albedo_tensor, 5).item(), 4)
    display_text = str(constants.IID_VERSION) + str(constants.ITERATION) + "<br>" \
                   "Mean Albedo PSNR, SSIM: <br>" \
                    "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                    "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                    "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                    "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>" + \
                    "Ours PSNR: " + str(psnr_albedo_e) + "<br> SSIM: " + str(ssim_albedo_e) + "<br>"

    visdom_reporter.plot_text(display_text)

    visdom_reporter.plot_image(albedo_tensor, "Albedo GT")
    visdom_reporter.plot_image(albedo_a_tensor, "Albedo li_eccv18")
    visdom_reporter.plot_image(albedo_b_tensor, "Albedo yu_cvpr19")
    visdom_reporter.plot_image(albedo_c_tensor, "Albedo yu_eccv20")
    visdom_reporter.plot_image(albedo_d_tensor, "Albedo zhu_iccp21")
    visdom_reporter.plot_image(albedo_e_tensor, "Albedo Ours")

    # psnr_a = np.round(kornia.metrics.psnr(shading_a_tensor, shading_tensor, max_val=1.0).item(), 4)
    # ssim_a = np.round(1.0 - kornia.losses.ssim_loss(shading_a_tensor, shading_tensor, 5).item(), 4)
    # psnr_b = np.round(kornia.metrics.psnr(shading_b_tensor, shading_tensor, max_val=1.0).item(), 4)
    # ssim_b = np.round(1.0 - kornia.losses.ssim_loss(shading_b_tensor, shading_tensor, 5).item(), 4)
    # psnr_c = np.round(kornia.metrics.psnr(shading_c_tensor, shading_tensor, max_val=1.0).item(), 4)
    # ssim_c = np.round(1.0 - kornia.losses.ssim_loss(shading_c_tensor, shading_tensor, 5).item(), 4)
    # psnr_d = np.round(kornia.metrics.psnr(shading_d_tensor, shading_tensor, max_val=1.0).item(), 4)
    # ssim_d = np.round(1.0 - kornia.losses.ssim_loss(shading_d_tensor, shading_tensor, 5).item(), 4)
    # display_text = "Mean Shading <br>" \
    #                              "li_eccv18 PSNR: " + str(psnr_a) + "<br> SSIM: " + str(ssim_a) + "<br>" \
    #                              "yu_cvpr19 PSNR: " + str(psnr_b) + "<br> SSIM: " + str(ssim_b) + "<br>" \
    #                              "yu_eccv20 PSNR: " + str(psnr_c) + "<br> SSIM: " + str(ssim_c) + "<br>" \
    #                              "zhu_iccp21 PSNR: " + str(psnr_d) + "<br> SSIM: " + str(ssim_d) + "<br>"
    #
    # visdom_reporter.plot_text(display_text)
    #
    # visdom_reporter.plot_image(shading_tensor, "Shading GT")
    # visdom_reporter.plot_image(shading_a_tensor, "Shading li_eccv18")
    # visdom_reporter.plot_image(shading_b_tensor, "Shading yu_cvpr19")
    # visdom_reporter.plot_image(shading_c_tensor, "Shading yu_eccv20")
    # visdom_reporter.plot_image(shading_d_tensor, "Shading zhu_iccp21")

    # psnr_a = np.round(kornia.metrics.psnr(rgb_a_tensor, rgb_tensor, max_val=1.0).item(), 4)
    # ssim_a = np.round(1.0 - kornia.losses.ssim_loss(rgb_a_tensor, rgb_tensor, 5).item(), 4)
    # psnr_b = np.round(kornia.metrics.psnr(rgb_b_tensor, rgb_tensor, max_val=1.0).item(), 4)
    # ssim_b = np.round(1.0 - kornia.losses.ssim_loss(rgb_b_tensor, rgb_tensor, 5).item(), 4)
    # psnr_c = np.round(kornia.metrics.psnr(rgb_c_tensor, rgb_tensor, max_val=1.0).item(), 4)
    # ssim_c = np.round(1.0 - kornia.losses.ssim_loss(rgb_c_tensor, rgb_tensor, 5).item(), 4)
    # psnr_d = np.round(kornia.metrics.psnr(rgb_d_tensor, rgb_tensor, max_val=1.0).item(), 4)
    # ssim_d = np.round(1.0 - kornia.losses.ssim_loss(rgb_d_tensor, rgb_tensor, 5).item(), 4)
    # display_text = "Image " + str(i) + " RGB <br>" \
    #                 "li_eccv18 PSNR: " + str(psnr_a) + "<br> SSIM: " + str(ssim_a) + "<br>" \
    #                 "yu_cvpr19 PSNR: " + str(psnr_b) + "<br> SSIM: " + str(ssim_b) + "<br>" \
    #                 "yu_eccv20 PSNR: " + str(psnr_c) + "<br> SSIM: " + str(ssim_c) + "<br>" \
    #                 "zhu_iccp21 PSNR: " + str(psnr_d) + "<br> SSIM: " + str(ssim_d) + "<br>"
    #
    # visdom_reporter.plot_text(display_text)
    #
    # visdom_reporter.plot_image(rgb_tensor, "RGB GT")
    # visdom_reporter.plot_image(rgb_a_tensor, "RGB li_eccv18")
    # visdom_reporter.plot_image(rgb_b_tensor, "RGB yu_cvpr19")
    # visdom_reporter.plot_image(rgb_c_tensor, "RGB yu_eccv20")
    # visdom_reporter.plot_image(rgb_d_tensor, "RGB zhu_iccp21")


def main(argv):
    (opts, args) = parser.parse_args(argv)
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

    img_list = glob.glob(opts.input_path + "*.jpg") + glob.glob(opts.input_path + "*.png")
    print("Images found: ", len(img_list))

    trainer = iid_trainer.IIDTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.rgb_l1_weight)

    constants.ITERATION = str(opts.iteration)
    constants.IID_VERSION = opts.version_name
    constants.IID_CHECKPATH = 'checkpoint/' + constants.IID_VERSION + "_" + constants.ITERATION + '.pt'
    checkpoint = torch.load(constants.IID_CHECKPATH, map_location=device)
    trainer.load_saved_state(checkpoint)

    albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
    rgb_dir_ws = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
    rgb_dir_ns = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
    constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
    print(rgb_dir_ws, albedo_dir)

    # Create the dataloader
    test_loader = dataset_loader.load_iid_datasetv2_test(rgb_dir_ws, rgb_dir_ns, albedo_dir, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    print("Plotting test images...")
    _, rgb_ws_batch, rgb_ns_batch, albedo_batch = next(iter(test_loader))
    rgb_ws_tensor = rgb_ws_batch.to(device)
    rgb_ns_tensor = rgb_ns_batch.to(device)
    albedo_tensor = albedo_batch.to(device)
    iid_op = iid_transforms.IIDTransform()
    input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

    trainer.visdom_visualize(rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Test")
    trainer.visdom_measure(rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Test")

    _, input_rgb_batch = next(iter(rw_loader))
    input_rgb_tensor = input_rgb_batch.to(device)
    trainer.visdom_infer(input_rgb_tensor)

    normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for i, input_path in enumerate(img_list, 0):
        filename = input_path.split("\\")[-1]
        input_tensor = tensor_utils.load_metric_compatible_img(input_path, cv2.COLOR_BGR2RGB, True, True, opts.img_size).to(device)
        input_tensor = normalize_op(input_tensor)

        albedo_tensor, shading_tensor, shadow_tensor = trainer.decompose(input_tensor)
        print(np.shape(albedo_tensor), np.shape(shading_tensor))

        albedo_tensor = albedo_tensor * 0.5 + 0.5

        vutils.save_image(albedo_tensor.squeeze(), opts.output_path + filename)

    measure_performance()


if __name__ == "__main__":
    main(sys.argv)
