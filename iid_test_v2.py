import glob
import sys
from optparse import OptionParser
import random
import cv2
import kornia
import torch
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
    def __init__(self, albedo_t, shading_t, shadow_t):
        print("Initiating")
        self.cgi_op = iid_transforms.CGITransform()
        self.iid_op = iid_transforms.IIDTransform()
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        # self.mask_t = mask_t
        self.albedo_t = albedo_t
        self.shading_t = shading_t
        self.shadow_t = shadow_t

        self.wdhr_metric_list = []

        self.psnr_list_rgb = []
        self.ssim_list_rgb = []

        self.psnr_list_eq = []
        self.ssim_list_eq = []

    def test_own_dataset(self, rgb_ws_tensor, rgb_ns_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, opts):
        input = {"rgb": rgb_ws_tensor, "unlit": unlit_tensor, "albedo": albedo_tensor}
        rgb2shading = self.shading_t.test(input)
        _, rgb2shadow = self.shadow_t.test(input)
        rgb2albedo = self.albedo_t.test(input)
        rgb_ns_like = self.iid_op.remove_rgb_shadow(rgb_ws_tensor, rgb2shadow, False)

        # normalize everything
        rgb_ws_tensor = tensor_utils.normalize_to_01(rgb_ws_tensor)
        rgb_ns_like = tensor_utils.normalize_to_01(rgb_ns_like)
        shading_tensor = tensor_utils.normalize_to_01(shading_tensor)
        shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)
        albedo_tensor = tensor_utils.normalize_to_01(albedo_tensor)
        rgb2shading = tensor_utils.normalize_to_01(rgb2shading)
        rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)

        rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow, False)

        self.visdom_reporter.plot_image(rgb_ws_tensor, "Input RGB Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns_like, "Input RGB (No Shadow) Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_like, "RGB Reconstruction Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(unlit_tensor, "Input Unlit Images - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(albedo_tensor, "GT Albedo - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2albedo, "RGB2Albedo - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(shading_tensor, "GT Shading - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2shading, "RGB2Shading - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(shadow_tensor, "GT Shadow - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2shadow, "RGB2Shadow - " + opts.version + str(opts.iteration))

        psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
        ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
        psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
        ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
        psnr_shadow = np.round(kornia.metrics.psnr(rgb2shadow, shadow_tensor, max_val=1.0).item(), 4)
        ssim_shadow = np.round(1.0 - kornia.losses.ssim_loss(rgb2shadow, shadow_tensor, 5).item(), 4)
        psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, rgb_ws_tensor, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, rgb_ws_tensor, 5).item(), 4)
        display_text = "Test Set - Versions: " + opts.version + "_" + str(opts.iteration) + \
                       "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) + \
                       "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                       "<br> Shadow PSNR: " + str(psnr_shadow) + "<br> Shadow SSIM: " + str(ssim_shadow) + \
                       "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)

        self.visdom_reporter.plot_text(display_text)

    def test_cgi(self, rgb_tensor, albedo_tensor, opts):
        rgb_tensor, albedo_tensor, shading_tensor = self.cgi_op.decompose_cgi(rgb_tensor, albedo_tensor)
        # albedo_inferred = self.cgi_op.extract_albedo(rgb_tensor, shading_tensor, torch.zeros_like(shading_tensor))
        # rgb_inferred = shading_tensor * albedo_inferred

        input = {"rgb": rgb_tensor}
        _, rgb2shadow = self.shadow_t.test(input)
        rgb_ns_like = self.iid_op.remove_rgb_shadow(rgb_tensor, rgb2shadow, False)

        # rgb2mask = self.mask_t.test(input)
        input = {"rgb": rgb_ns_like}
        rgb2albedo = self.albedo_t.test(input)
        rgb2shading = self.shading_t.test(input)

        # normalize everything
        shading_tensor = tensor_utils.normalize_to_01(shading_tensor)
        albedo_tensor = tensor_utils.normalize_to_01(albedo_tensor)
        rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
        rgb2shading = tensor_utils.normalize_to_01(rgb2shading)
        rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
        rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow, False)
        rgb_ns_like = tensor_utils.normalize_to_01(rgb_ns_like)

        self.visdom_reporter.plot_image(rgb_tensor, "CGI RGB Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns_like, "CGI RGB (No Shadows) Images - " + opts.version + str(opts.iteration))
        # self.visdom_reporter.plot_image(rgb_inferred, "CGI RGB Inferred - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_like, "CGI RGB Like - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(albedo_tensor, "CGI Albedo Images - " + opts.version + str(opts.iteration))
        # self.visdom_reporter.plot_image(albedo_inferred, "CGI Albedo Inferred - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2albedo, "CGI Albedo Like - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(shading_tensor, "CGI Shading Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2shading, "CGI Shading Like - " + opts.version + str(opts.iteration))

        self.visdom_reporter.plot_image(rgb2shadow, "CGI Shadow Like - " + opts.version + str(opts.iteration))

        psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
        ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
        psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
        ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
        psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, rgb_tensor, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, rgb_tensor, 5).item(), 4)
        display_text = "CGI Set - Versions: " + opts.version + "_" + str(opts.iteration) + \
                       "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) + \
                       "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                       "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)

        self.visdom_reporter.plot_text(display_text)

    def test_shadow(self, rgb_ws, rgb_ns, prefix, opts):
        rgb_ws = torch.clip(rgb_ws, 0.0, 1.0)
        rgb_ns = torch.clip(rgb_ns, 0.0, 1.0)

        # input = {"rgb": rgb_tensor_ws}
        # rgb2ns_img, rgb2shadow = self.shadow_t.test(input)
        # rgb2ns_eq = self.iid_op.remove_rgb_shadow(rgb_tensor_ws, rgb2shadow, False)

        rgb_ws, rgb_ns, shadow_matte, rgb_ws_relit, rgb_ns_eq = self.iid_op.decompose_shadow(rgb_ws, rgb_ns)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        shadow_matte = tensor_utils.normalize_to_01(shadow_matte)
        rgb_ws_relit = tensor_utils.normalize_to_01(rgb_ws_relit)
        rgb_ns_eq = tensor_utils.normalize_to_01(rgb_ns_eq)

        self.visdom_reporter.plot_image(rgb_ws, prefix + " WS Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ws_relit, prefix + " Relit Images - " + opts.version + str(opts.iteration))
        # self.visdom_reporter.plot_image(rgb2ns_eq, prefix + " NS-Like (Equation) Images - " + opts.version + str(opts.iteration))
        # self.visdom_reporter.plot_image(rgb2ns_img, prefix + " NS-Like (Generated) Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns, prefix + " NS Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns_eq, prefix + " NS (equation) Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(shadow_matte, prefix + " Shadow Matte - " + opts.version + str(opts.iteration))

        psnr_eq = np.round(kornia.metrics.psnr(rgb_ns_eq, rgb_ns, max_val=1.0).item(), 4)
        ssim_eq = np.round(1.0 - kornia.losses.ssim_loss(rgb_ns_eq, rgb_ns, 5).item(), 4)
        self.psnr_list_eq.append(psnr_eq)
        self.ssim_list_eq.append(ssim_eq)

    def print_ave_shadow_performance(self, prefix, opts):
        ave_psnr_rgb = np.round(np.mean(self.psnr_list_rgb), 4)
        ave_ssim_rgb = np.round(np.mean(self.ssim_list_rgb), 4)
        ave_psnr_eq = np.round(np.mean(self.psnr_list_eq), 4)
        ave_ssim_eq = np.round(np.mean(self.ssim_list_eq), 4)
        display_text = prefix + " - Versions: " + opts.version + "_" + str(opts.iteration) + \
                       "<br> EQ Reconstruction PSNR: " + str(ave_psnr_eq) + "<br> EQ Reconstruction SSIM: " + str(ave_ssim_eq) + \
                       "<br> RGB Reconstruction PSNR: " + str(ave_psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ave_ssim_rgb)

        self.visdom_reporter.plot_text(display_text)

        self.psnr_list_rgb.clear()
        self.ssim_list_rgb.clear()

    def test_iiw(self, file_name, rgb_tensor, opts):
        input = {"rgb": rgb_tensor}
        # rgb2mask = self.mask_t.test(input)
        _, rgb2shadow = self.shadow_t.test(input)
        rgb_ns_like = self.iid_op.remove_rgb_shadow(rgb_tensor, rgb2shadow, False)

        input = {"rgb" : rgb_ns_like}
        rgb2albedo = self.albedo_t.test(input)
        rgb2shading = self.shading_t.test(input)

        # normalize everything
        rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
        rgb2shading = tensor_utils.normalize_to_01(rgb2shading)
        rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
        rgb_ns_like = tensor_utils.normalize_to_01(rgb_ns_like)

        rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow, False)

        self.visdom_reporter.plot_image(rgb_tensor, "IIW RGB Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns_like, "IIW RGB (No Shadow) Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2albedo, "IIW Albedo Images - " + opts.version + str(opts.iteration))

        judgements_dir = "E:/iiw-decompositions/iiw-dataset/data/"
        # print(np.shape(rgb2albedo))
        for i in range(np.shape(rgb2albedo)[0]):
            img_file_name = "./iiw_temp/" + file_name[i] + ".png"
            torchvision.utils.save_image(rgb2albedo[i], img_file_name)

            whdr_metric = whdr.whdr_final(img_file_name, judgements_dir + file_name[i] + ".json")
            self.wdhr_metric_list.append(whdr_metric)

    def test_rw(self, rgb_tensor, opts):
        input = {"rgb": rgb_tensor}
        # rgb2mask = self.mask_t.test(input)
        rgb2albedo = self.albedo_t.test(input)
        rgb2shading = self.shading_t.test(input)
        _, rgb2shadow = self.shadow_t.test(input)
        rgb_ns_like = self.iid_op.remove_rgb_shadow(rgb_tensor, rgb2shadow, False)

        # normalize everything
        rgb_tensor = tensor_utils.normalize_to_01(rgb_tensor)
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
        # rgb2albedo = rgb2albedo * rgb2mask
        # rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)
        rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)
        rgb_ns_like = tensor_utils.normalize_to_01(rgb_ns_like)

        self.visdom_reporter.plot_image(rgb_tensor, "RW RGB Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_ns_like, "RW RGB (No Shadow) Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb_like, "RW RGB Reconstruction Images - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2albedo, "RW RGB2Albedo - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2shading, "RW RGB2Shading - " + opts.version + str(opts.iteration))
        self.visdom_reporter.plot_image(rgb2shadow, "RW RGB2Shadow - " + opts.version + str(opts.iteration))

    def test_gta(self, opts):
        img_list = glob.glob(opts.input_path + "*.jpg") + glob.glob(opts.input_path + "*.png")
        print("Images found: ", len(img_list))

        normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")

        for i, input_path in enumerate(img_list, 0):
            filename = input_path.split("\\")[-1]
            input_rgb_tensor = tensor_utils.load_metric_compatible_img(input_path, cv2.COLOR_BGR2RGB, True, True, opts.img_size).to(device)
            input_rgb_tensor = normalize_op(input_rgb_tensor)
            input = {"rgb": input_rgb_tensor}
            _, rgb2shadow = self.shadow_t.test(input)

            rgb_ns_like = self.iid_op.remove_rgb_shadow(input_rgb_tensor, rgb2shadow, False)

            # rgb2mask = mask_t.test(input)
            input = {"rgb": rgb_ns_like}
            rgb2albedo = self.albedo_t.test(input)
            rgb2shading = self.shading_t.test(input)

            # normalize everything
            rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)
            # rgb2albedo = rgb2albedo * rgb2mask
            # rgb2albedo = iid_op.mask_fill_nonzeros(rgb2albedo)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading, rgb2shadow)
            rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
            input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
            rgb_ns_like = tensor_utils.normalize_to_01(rgb_ns_like)

            self.visdom_reporter.plot_image(rgb2shadow, "GTA Shadow Maps - " + opts.version + str(opts.iteration) + str(i))
            self.visdom_reporter.plot_image(input_rgb_tensor, "GTA RGB (Original) Images - " + opts.version + str(opts.iteration) + str(i))
            self.visdom_reporter.plot_image(rgb_ns_like, "GTA RGB (No Shadow) Images - " + opts.version + str(opts.iteration) + str(i))

            vutils.save_image(rgb2albedo.squeeze(), opts.output_path + filename)

    def get_average_whdr(self, opts):
        ave_whdr = np.round(np.mean(self.wdhr_metric_list), 4)
        display_text = "IIW Average WHDR for Version: " + opts.version + "_" + str(opts.iteration) + \
        "<br> Average WHDR: " + str(ave_whdr)

        self.visdom_reporter.plot_text(display_text)


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
    network_config = sc_instance.interpret_network_config_from_version(opts.version)
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