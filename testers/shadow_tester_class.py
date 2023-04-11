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
import yaml
from torchvision.transforms import transforms
from yaml import SafeLoader
from config.network_config import ConfigHolder
from loaders import dataset_loader
from transforms import iid_transforms
import global_config
from utils import plot_utils, tensor_utils
from trainers import trainer_factory
from losses import whdr
import torchvision.transforms.functional

class TesterClass():
    def __init__(self, shadow_m, shadow_t):
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.shadow_m = shadow_m
        self.shadow_t = shadow_t

        self.wdhr_metric_list = []

        self.psnr_list_rgb = []
        self.ssim_list_rgb = []
        self.mae_list_rgb = []

        self.rmse_list_lab = []
        self.rmse_list_lab_ws = []

        self.mae_list_sm = []
        self.mae_list_sm_ws = []


    def infer_shadow_results(self, rgb_ws, shadow_matte, mode):
        if (mode == "train_shadow"):
            # only test shadow removal
            input_map = {"rgb": rgb_ws, "shadow_matte": shadow_matte}
            rgb2ns = self.shadow_t.test(input_map)
            rgb2sm = None
        else:
            # test shadow matte inference + shadow removal
            rgb2sm = self.shadow_m.test({"rgb": rgb_ws})
            input_map = {"rgb": rgb_ws, "shadow_matte": rgb2sm}
            rgb2ns = self.shadow_t.test(input_map)

        return rgb2ns, rgb2sm

    def test_shadow_matte(self, file_name, rgb_ws, shadow_matte, prefix, show_images, save_image_results, opts):
        device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
        rgb2sm = self.shadow_m.test({"rgb": rgb_ws})

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        shadow_matte = tensor_utils.normalize_to_01(shadow_matte)
        rgb2sm = tensor_utils.normalize_to_01(rgb2sm)

        if (show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, prefix + " WS Images - " + global_config.sm_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(shadow_matte, prefix + " WS Shadow Matte Images - " + global_config.sm_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb2sm, prefix + " WS Shadow Matte-Like Images - " + global_config.sm_network_version + str(global_config.sm_iteration))

        if (save_image_results == 1):
            if(prefix == "ISTD"):
                path = "./comparison/ISTD Dataset/OURS/"
            else:
                path = "./comparison/SRD Dataset/OURS/"
            matte_path = path + "/matte-like/"
            gt_path = path + "/matte/"
            for i in range(0, np.size(file_name)):
                shadow_matte_path = matte_path + file_name[i] + ".png"
                torchvision.utils.save_image(rgb2sm[i], shadow_matte_path, normalize=True)

                gt_matte_path = gt_path + file_name[i] + ".png"
                torchvision.utils.save_image(shadow_matte[i], gt_matte_path, normalize=True)

                # gt_matte_path = gt_path + file_name[i] + ".png"
                # torchvision.utils.save_image(rgb_ws[i], gt_matte_path, normalize=False)

        mae = nn.L1Loss()
        mse = nn.MSELoss()

        # shadow matte
        mae_sm = np.round(mae(rgb2sm, shadow_matte).cpu(), 4)
        self.mae_list_sm.append(mae_sm)

        rmse_sm = np.round(mse(rgb2sm, shadow_matte).cpu(), 4)
        self.rmse_list_lab.append(torch.sqrt(rmse_sm))

        if(prefix == "ISTD"):
            input_shape = np.shape(rgb2sm[0])
            transform_op = transforms.Compose([transforms.ToPILImage(), transforms.Resize((input_shape[1], input_shape[2])), transforms.ToTensor()])
            mask_path = "E:/ISTD_Dataset/test/test_B/"

            for i in range(0, np.size(file_name)):
                shadow_mask = transform_op(cv2.cvtColor(cv2.imread(mask_path + file_name[i] + ".png"), cv2.COLOR_BGR2GRAY))
                shadow_mask = shadow_mask.to(device)
                # print("Shapes: ", np.shape(rgb2sm[i]), np.shape(shadow_mask), np.shape(shadow_matte[i]))
                mae_sm_ws = np.round(mae(rgb2sm[i] * shadow_mask, shadow_matte[i] * shadow_mask).cpu(), 4)
                self.mae_list_sm_ws.append(mae_sm_ws)

                rmse_sm_ws = np.round(mse(rgb2sm[i] * shadow_mask, shadow_matte[i] * shadow_mask).cpu(), 4)
                self.rmse_list_lab_ws.append(torch.sqrt(rmse_sm_ws))

        elif (prefix == "SRD"):
            input_shape = np.shape(rgb2sm[0])
            transform_op = transforms.Compose([transforms.ToPILImage(), transforms.Resize((input_shape[1], input_shape[2])), transforms.ToTensor()])
            mask_path = "E:/SRD_Test/srd/mask/"

            for i in range(0, np.size(file_name)):
                shadow_mask = cv2.imread(mask_path + file_name[i])
                if (shadow_mask is not None):
                    shadow_mask = transform_op(cv2.cvtColor(cv2.imread(mask_path + file_name[i]), cv2.COLOR_BGR2GRAY))
                    shadow_mask = shadow_mask.to(device)
                    # print("Shapes: ", np.shape(rgb2sm[i]), np.shape(shadow_mask), np.shape(shadow_matte[i]))
                    mae_sm_ws = np.round(mae(rgb2sm[i] * shadow_mask, shadow_matte[i] * shadow_mask).cpu(), 4)
                    self.mae_list_sm_ws.append(mae_sm_ws)

                    rmse_sm_ws = np.round(mse(rgb2sm[i] * shadow_mask, shadow_matte[i] * shadow_mask).cpu(), 4)
                    self.rmse_list_lab_ws.append(torch.sqrt(rmse_sm_ws))

    def print_shadow_matte_performance(self, prefix, opts):
        ave_mae_sm = np.round(np.mean(self.mae_list_sm) * 100.0, 4)
        ave_mae_sm_ws = np.round(np.mean(self.mae_list_sm_ws) * 100.0, 4)

        ave_rmse_sm = np.round(np.mean(self.rmse_list_lab) * 100.0, 4)
        ave_rmse_sm_ws = np.round(np.mean(self.rmse_list_lab_ws) * 100.0, 4)

        network_config = iid_server_config.IIDServerConfig.getInstance().interpret_shadow_matte_params_from_version()

        display_text = prefix + " - Versions: " + global_config.sm_network_version + "_" + str(global_config.sm_iteration) + \
                       "<br>" + network_config["dataset_version"] + \
                       "<br> MAE Error (SM): " + str(ave_mae_sm) + "<br> MAE Error (SM WS): " + str(ave_mae_sm_ws) + \
                       "<br> RMSE Error (SM): " + str(ave_rmse_sm) + "<br> RMSE Error (SM WS): " + str(ave_rmse_sm_ws)

        self.visdom_reporter.plot_text(display_text)

        self.mae_list_sm.clear()
        self.mae_list_sm_ws.clear()

        self.rmse_list_lab.clear()
        self.rmse_list_lab_ws.clear()

    def test_any_image(self, file_name, rgb_ws, prefix, show_images, save_image_results, opts):
        device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
        rgb2sm = self.shadow_m.test({"rgb": rgb_ws})
        input_map = {"rgb": rgb_ws, "shadow_matte": rgb2sm}
        rgb2ns = self.shadow_t.test(input_map)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)

        if (show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, prefix + " WS Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb2ns, prefix + " NS (equation) Images - " + global_config.ns_network_version + str(global_config.sm_iteration))

        if (save_image_results == 1):
            path = "./comparison/"
            matte_path = path + "/matte-like/"

            for i in range(0, np.size(file_name)):
                impath = path + file_name[i]
                torchvision.utils.save_image(rgb2ns[i], impath)

                # shadow_matte_path = matte_path + file_name[i] + ".jpeg"
                # torchvision.utils.save_image(shadow_matte[i], shadow_matte_path)

                # print("Saving ISTD result as: ", file_name[i])


    def test_shadow(self, rgb_ws, rgb_ns, shadow_matte, prefix, show_images, mode):
        rgb2ns, rgb2sm = self.infer_shadow_results(rgb_ws, shadow_matte, mode)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)
        # rgb2sm = tensor_utils.normalize_to_01(rgb2sm)

        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, prefix + " WS Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            # self.visdom_reporter.plot_image(shadow_matte, "WS Shadow Matte Images - " + opts.shadow_network_version + str(opts.iteration))
            # if(rgb2sm != None):
            #     rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
            #     self.visdom_reporter.plot_image(rgb2sm, prefix + " WS Shadow Matte-Like Images - " + opts.shadow_network_version + str(opts.iteration))
            self.visdom_reporter.plot_image(rgb_ns, prefix + " NS Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb2ns, prefix + " NS (equation) Images - " + global_config.ns_network_version + str(global_config.sm_iteration))

        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae = nn.L1Loss()
        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)

        mse = nn.MSELoss()
        mse_lab = np.round(mse(kornia.color.rgb_to_lab(rgb2ns), kornia.color.rgb_to_lab(rgb_ns)).cpu(), 4)
        rmse_lab = torch.sqrt(mse_lab)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)
        self.rmse_list_lab.append(rmse_lab)

    #for ISTD
    def test_istd_shadow(self, file_name, rgb_ws, rgb_ns, shadow_matte, show_images, save_image_results, mode):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        ### NOTE: ISTD-NS (No Shadows) image already has a different lighting!!! This isn't reported in the dataset. Consider using ISTD-NS as the unmasked region to avoid bias in results.
        ### MAE discrepancy vs ISTD-WS is at 11.055!
        rgb2ns, rgb2sm = self.infer_shadow_results(rgb_ws, shadow_matte, mode)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)

        resize_op = transforms.Resize((240, 320), transforms.InterpolationMode.BICUBIC)
        rgb_ns = resize_op(rgb_ws)
        rgb2ns = resize_op(rgb2ns)

        if(rgb2sm != None):
            rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
            shadow_matte = tensor_utils.normalize_to_01(shadow_matte)

        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, "ISTD WS Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(shadow_matte, "ISTD Shadow Matte Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
            if (rgb2sm != None):
                self.visdom_reporter.plot_image(rgb2sm, "ISTD Shadow Matte-Like Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb_ns, "ISTD NS Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb2ns, "ISTD NS-Like Images - " + global_config.sm_network_version + str(global_config.sm_iteration) + " " + global_config.ns_network_version + str(global_config.sm_iteration))

        mae = nn.L1Loss()
        mse = nn.MSELoss()
        if(save_image_results == 1):
            path = "./comparison/ISTD Dataset/OURS/"
            matte_path = path + "/matte-like/"

            input_shape = np.shape(rgb2ns[0])
            transform_op = transforms.Compose([transforms.ToPILImage(), transforms.Resize((input_shape[1], input_shape[2])), transforms.ToTensor()])
            mask_path = "E:/ISTD_Dataset/test/test_B/"

            for i in range(0, np.size(file_name)):
                impath = path + file_name[i] + ".png"
                torchvision.utils.save_image(rgb2ns[i], impath)

                # shadow_matte_path = matte_path + file_name[i] + ".jpeg"
                # torchvision.utils.save_image(shadow_matte[i], shadow_matte_path)

                # print("Saving ISTD result as: ", file_name[i])

                shadow_mask = transform_op(cv2.cvtColor(cv2.imread(mask_path + file_name[i] + ".png"), cv2.COLOR_BGR2GRAY))
                shadow_mask = shadow_mask.to(device)
                # print("Shapes: ", np.shape(rgb2sm[i]), np.shape(shadow_mask), np.shape(shadow_matte[i]))
                rmse_lab_ws = np.round(mse(kornia.color.rgb_to_lab(rgb2ns * shadow_mask), kornia.color.rgb_to_lab(rgb_ns * shadow_mask)).cpu(), 4)
                self.rmse_list_lab_ws.append(torch.sqrt(rmse_lab_ws))


        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)
        mse_lab = np.round(mse(kornia.color.rgb_to_lab(rgb2ns), kornia.color.rgb_to_lab(rgb_ns)).cpu(), 4)
        rmse_lab = torch.sqrt(mse_lab)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)
        self.rmse_list_lab.append(rmse_lab)

    def test_srd(self, file_name, rgb_ws, rgb_ns, shadow_matte, show_images, save_image_results, mode):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        rgb2ns, rgb2sm = self.infer_shadow_results(rgb_ws, shadow_matte, mode)

        # normalize everything
        rgb_ws = tensor_utils.normalize_to_01(rgb_ws)
        rgb_ns = tensor_utils.normalize_to_01(rgb_ns)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns = torch.clip(rgb2ns, 0.0, 1.0)

        resize_op = transforms.Resize((160, 210), transforms.InterpolationMode.BICUBIC)
        rgb_ns = resize_op(rgb_ws)
        rgb2ns = resize_op(rgb2ns)

        if (rgb2sm != None):
            rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
            shadow_matte = tensor_utils.normalize_to_01(shadow_matte)

        if(show_images == 1):
            self.visdom_reporter.plot_image(rgb_ws, "SRD WS Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(shadow_matte, "SRD Shadow Matte Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            if (rgb2sm != None):
                self.visdom_reporter.plot_image(rgb2sm, "SRD Shadow Matte-Like Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb_ns, "SRD NS Images - " + global_config.ns_network_version + str(global_config.sm_iteration))
            self.visdom_reporter.plot_image(rgb2ns, "SRD NS-Like Images - " + global_config.ns_network_version + str(global_config.sm_iteration))

        mae = nn.L1Loss()
        mse = nn.MSELoss()

        if(save_image_results == 1):
            path = "./comparison/SRD Dataset/OURS/"
            matte_path = path + "/matte-like/"

            input_shape = np.shape(rgb2ns[0])
            transform_op = transforms.Compose([transforms.ToPILImage(), transforms.Resize((input_shape[1], input_shape[2])), transforms.ToTensor()])
            mask_path = "E:/SRD_Test/srd/mask/"

            for i in range(0, np.size(file_name)):
                impath = path + file_name[i] + ".png"
                torchvision.utils.save_image(rgb2ns[i], impath)

                # shadow_matte_path = matte_path + file_name[i]
                # torchvision.utils.save_image(shadow_matte[i], shadow_matte_path)

                shadow_mask = cv2.imread(mask_path + file_name[i] + ".jpg")
                if(shadow_mask is not None):
                    # print("Mask path: " + (mask_path + file_name[i] + ".jpg"))
                    shadow_mask = transform_op(cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY))
                    shadow_mask = shadow_mask.to(device)
                    # print("Shapes: ", np.shape(rgb2sm[i]), np.shape(shadow_mask), np.shape(shadow_matte[i]))
                    rmse_lab_ws = np.round(mse(kornia.color.rgb_to_lab(rgb2ns * shadow_mask), kornia.color.rgb_to_lab(rgb_ns * shadow_mask)).cpu(), 4)
                    self.rmse_list_lab_ws.append(torch.sqrt(rmse_lab_ws))

        psnr_rgb = np.round(kornia.metrics.psnr(rgb2ns, rgb_ns, max_val=1.0).item(), 4)
        ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb2ns, rgb_ns, 5).item(), 4)

        mae_rgb = np.round(mae(rgb2ns, rgb_ns).cpu(), 4)
        mse_lab = np.round(mse(kornia.color.rgb_to_lab(rgb2ns), kornia.color.rgb_to_lab(rgb_ns)).cpu(), 4)
        rmse_lab = torch.sqrt(mse_lab)

        self.psnr_list_rgb.append(psnr_rgb)
        self.ssim_list_rgb.append(ssim_rgb)
        self.mae_list_rgb.append(mae_rgb)
        self.rmse_list_lab.append(rmse_lab)

    def print_ave_shadow_performance(self, prefix):
        ave_psnr_rgb = np.round(np.mean(self.psnr_list_rgb), 4)
        ave_ssim_rgb = np.round(np.mean(self.ssim_list_rgb), 4)
        ave_mae_rgb = np.round(np.mean(self.mae_list_rgb) * 255.0, 4)
        ave_mae_sm = np.round(np.mean(self.mae_list_sm) * 255.0, 4)
        ave_rmse_lab = np.round(np.mean(self.rmse_list_lab), 4)
        ave_rmse_lab_ws = np.round(np.mean(self.rmse_list_lab_ws), 4)

        display_text = prefix + " - Versions: " + global_config.sm_network_version + "_" + str(global_config.sm_iteration) + \
                       "<br>" + global_config.ns_network_version + "_" + str(global_config.ns_iteration) + \
                       "<br> MAE Error (SM): " + str(ave_mae_sm) + "<br> MAE Error (RGB): " +str(ave_mae_rgb) + \
                       "<br> RGB Reconstruction PSNR: " + str(ave_psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ave_ssim_rgb) + \
                       "<br> Lab RMSE: " + str(ave_rmse_lab) + "<br> Lab RMSE WS: " +str(ave_rmse_lab_ws)

        self.visdom_reporter.plot_text(display_text)

        self.psnr_list_rgb.clear()
        self.ssim_list_rgb.clear()
        self.mae_list_rgb.clear()
        self.mae_list_sm.clear()
        self.rmse_list_lab_ws.clear()
        self.rmse_list_lab.clear()