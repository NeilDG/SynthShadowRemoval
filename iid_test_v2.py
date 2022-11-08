import glob
import sys
from optparse import OptionParser
import random
import cv2
import kornia
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils
import torchvision.utils as vutils
import numpy as np
from torchvision.transforms import transforms
from config import iid_server_config
from loaders import dataset_loader
from transforms import iid_transforms
import constants
from utils import plot_utils, tensor_utils
from trainers import trainer_factory
from custom_losses import whdr
import torchvision.transforms.functional

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
parser.add_option('--input_path', type=str)
parser.add_option('--output_path', type=str)
parser.add_option('--img_size', type=int, default=(256, 256))

def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/unlit/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
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
        constants.DATASET_PLACES_PATH = "D:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "D:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "D:/Datasets/SynthWeather Dataset 8/albedo/"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers)
    else:
        opts.num_workers = 12
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws_styled = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns_styled = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "E:/SynthWeather Dataset 8/unlit/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers)

def measure_performance(opts):
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

    albedo_tensor = None
    albedo_a_tensor = None
    albedo_b_tensor = None
    albedo_c_tensor = None
    albedo_d_tensor = None
    albedo_e_tensor = None


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
    display_text = "GTA Performance " + str(opts.version) + str(opts.iteration) + "<br>" \
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

class TesterClass():
    def __init__(self, shadow_t):
        print("Initiating")
        self.cgi_op = iid_transforms.CGITransform()
        self.iid_op = iid_transforms.IIDTransform()
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.shadow_t = shadow_t

        self.wdhr_metric_list = []

        self.psnr_list_rgb = []
        self.ssim_list_rgb = []
        self.mae_list_rgb = []

        self.mae_list_sm = []

    def infer_shadow_results(self, rgb_ws, shadow_matte):
        input_map = {"rgb": rgb_ws, "shadow_matte": shadow_matte}
        rgb2ns = self.shadow_t.test(input_map)

        return rgb2ns


    def test_shadow(self, rgb_ws, rgb_ns, shadow_matte, prefix, show_images, opts):
        rgb2ns = self.infer_shadow_results(rgb_ws, shadow_matte)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)
        # rgb2sm = tensor_utils.normalize_to_01(rgb2sm)

        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, prefix + " WS Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb_ns, prefix + " NS Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb2ns, prefix + " NS (equation) Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))

        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae = nn.L1Loss()
        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)


    #for ISTD
    def test_istd_shadow(self, file_name, rgb_ws, rgb_ns, shadow_matte, show_images, save_image_results, opts):
        ### NOTE: ISTD-NS (No Shadows) image already has a different lighting!!! This isn't reported in the dataset. Consider using ISTD-NS as the unmasked region to avoid bias in results.
        ### MAE discrepancy vs ISTD-WS is at 11.055!
        rgb2ns = self.infer_shadow_results(rgb_ws, shadow_matte)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)

        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, "ISTD WS Images - " + opts.shadow_matte_network_version + str(opts.shadow_matte_iteration) + " " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(shadow_matte, "ISTD Shadow Matte Images - " + opts.shadow_matte_network_version + str(opts.shadow_matte_iteration) + " " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb_ns, "ISTD NS Images - " + opts.shadow_matte_network_version + str(opts.shadow_matte_iteration) + " " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb2ns, "ISTD NS-Like Images - " + opts.shadow_matte_network_version + str(opts.shadow_matte_iteration) + " " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))

        if(save_image_results == 1):
            path = "./comparison/ISTD Dataset/OURS/"
            matte_path = path + "/matte/"
            for i in range(0, np.size(file_name)):
                impath = path + file_name[i] + ".png"
                torchvision.utils.save_image(rgb2ns[i], impath)


        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae = nn.L1Loss()
        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)

    def test_srd(self, file_name, rgb_ws, rgb_ns, shadow_matte, show_images, save_image_results, opts):
        rgb2ns = self.infer_shadow_results(rgb_ws, shadow_matte)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)
        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, "SRD WS Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(shadow_matte, "SRD Shadow Matte Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb_ns, "SRD NS Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))
            self.visdom_reporter.plot_image(rgb2ns, "SRD NS-Like Images - " + opts.shadow_removal_version + str(opts.shadow_removal_iteration))

        if(save_image_results == 1):
            path = "./comparison/SRD Dataset/OURS/"
            matte_path = path + "/matte-like/"
            for i in range(0, np.size(file_name)):
                impath = path + file_name[i] + ".png"
                torchvision.utils.save_image(rgb2ns[i], impath)

        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae = nn.L1Loss()
        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)

    def print_ave_shadow_performance(self, prefix, opts):
        ave_psnr_rgb = np.round(np.mean(self.psnr_list_rgb), 4)
        ave_ssim_rgb = np.round(np.mean(self.ssim_list_rgb), 4)
        ave_mae_rgb = np.round(np.mean(self.mae_list_rgb) * 255.0, 4)

        ave_mae_sm = np.round(np.mean(self.mae_list_sm) * 255.0, 4)

        display_text = prefix + " - Versions: " + opts.shadow_matte_network_version + "_" + str(opts.shadow_matte_iteration) + \
                       "<br>" + opts.shadow_removal_version + "_" + str(opts.shadow_removal_iteration) + \
                       "<br> MAE Error (SM): " + str(ave_mae_sm) + "<br> MAE Error (RGB): " +str(ave_mae_rgb) + \
                       "<br> RGB Reconstruction PSNR: " + str(ave_psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ave_ssim_rgb)

        self.visdom_reporter.plot_text(display_text)

        self.psnr_list_rgb.clear()
        self.ssim_list_rgb.clear()
        self.mae_list_rgb.clear()
        self.mae_list_sm.clear()


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

    print(constants.rgb_dir_ws_styled, constants.albedo_dir)
    plot_utils.VisdomReporter.initialize()
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    iid_server_config.IIDServerConfig.initialize(opts.version)
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_shadow_network_params_from_version(opts.version)
    print("General config:", general_config)
    print("Network config: ", network_config)

    iid_op = iid_transforms.IIDTransform()

    style_enabled = network_config["style_transferred"]
    if (style_enabled == 1):
        rgb_dir_ws = constants.rgb_dir_ws_styled
        rgb_dir_ns = constants.rgb_dir_ns_styled
    else:
        rgb_dir_ws = constants.rgb_dir_ws
        rgb_dir_ns = constants.rgb_dir_ns

    cgi_rgb_dir = "E:/CGIntrinsics/images/*/*_mlt.png"
    train_loader = dataset_loader.load_iid_datasetv2_test(rgb_dir_ws, rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, 256, opts)

    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    tf = trainer_factory.TrainerFactory(device, opts)
    mask_t, albedo_t, shading_t, shadow_t = tf.get_all_trainers()
    dataset_tester = TesterClass(mask_t, albedo_t, shading_t, shadow_t)

    for i, (_, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch) in enumerate(train_loader, 0):
        with torch.no_grad():
            rgb_ws_tensor = rgb_ws_batch.to(device)
            rgb_ns_tensor = rgb_ns_batch.to(device)
            albedo_tensor = albedo_batch.to(device)
            unlit_tensor = unlit_batch.to(device)
            rgb_ws_tensor, rgb_ns_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)
            dataset_tester.test_own_dataset(rgb_ws_tensor, rgb_ns_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, opts)
            break

    test_loader = dataset_loader.load_cgi_dataset(cgi_rgb_dir, 480, opts)
    for i, (file_name, rgb_batch, albedo_batch) in enumerate(test_loader, 0):
        # CGI dataset
        rgb_tensor = rgb_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        dataset_tester.test_cgi(rgb_tensor, albedo_tensor, opts)
        break

    #IIW dataset
    iiw_rgb_dir = "E:/iiw-decompositions/original_image/*.jpg"
    test_loader = dataset_loader.load_iiw_dataset(iiw_rgb_dir, opts)
    for i, (file_name, rgb_img) in enumerate(test_loader, 0):
        with torch.no_grad():
            rgb_tensor = rgb_img.to(device)
            dataset_tester.test_iiw(file_name, rgb_tensor, opts)

    dataset_tester.get_average_whdr(opts)

    #check RW performance
    _, input_rgb_batch = next(iter(rw_loader))
    input_rgb_tensor = input_rgb_batch.to(device)
    rgb2unlit = tf.get_unlit_network()(input_rgb_tensor)
    dataset_tester.test_rw(input_rgb_tensor, rgb2unlit, opts)

    #measure GTA performance
    img_list = glob.glob(opts.input_path + "*.jpg") + glob.glob(opts.input_path + "*.png")
    print("Images found: ", len(img_list))

    normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for i, input_path in enumerate(img_list, 0):
        filename = input_path.split("\\")[-1]
        input_rgb_tensor = tensor_utils.load_metric_compatible_img(input_path, cv2.COLOR_BGR2RGB, True, True, opts.img_size).to(device)
        input_rgb_tensor = normalize_op(input_rgb_tensor)
        rgb2unlit = tf.get_unlit_network()(input_rgb_tensor)

        input = {"rgb": input_rgb_tensor, "unlit": rgb2unlit}
        rgb2mask = mask_t.test(input)
        rgb2albedo = albedo_t.test(input)
        rgb2shading = shading_t.test(input)
        _, rgb2shadow = shadow_t.test(input)

        # normalize everything
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
        # rgb2albedo = rgb2albedo * rgb2mask
        # rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)
        rgb_like = iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)

        vutils.save_image(rgb2albedo.squeeze(), opts.output_path + filename)

    measure_performance(opts)


if __name__ == "__main__":
    main(sys.argv)