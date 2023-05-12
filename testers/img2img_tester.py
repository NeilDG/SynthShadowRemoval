import os.path

import kornia.metrics.psnr
import torchvision

from config import network_config
from config.network_config import ConfigHolder
import global_config
import torch
from utils import plot_utils, tensor_utils
import lpips
import torch.nn as nn
import numpy as np
from trainers import img2imgtrainer

class Img2ImgTester():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.img2img_t = img2imgtrainer.Img2ImgTrainer(self.gpu_device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.mse_results = []
        self.psnr_results = []

    def save_images(self, input_map_a, input_map_b):
        img_a2b, img_b2a = self.img2img_t.test(input_map_a)  # a2b --> real2synth, b2a --> synth2real
        file_name = input_map_a["file_name"]

        noshadows_path = "X:/GithubProjects/NeuralNets-Experiment3/reports/Synth2Real/rgb_noshadows/"
        withshadows_path = "X:/GithubProjects/NeuralNets-Experiment3/reports/Synth2Real/rgb/"

        if not os.path.exists(noshadows_path):
            os.makedirs(noshadows_path, exist_ok=True)

        if not os.path.exists(withshadows_path):
            os.makedirs(withshadows_path, exist_ok=True)

        for i in range(0, len(file_name)):
            impath = noshadows_path + file_name[i] + ".png"
            torchvision.utils.save_image(img_b2a[i], impath, normalize = True)
            # print("Saved image (no shadows) : ", impath)

        img_a2b, img_b2a = self.img2img_t.test(input_map_b)  # a2b --> real2synth, b2a --> synth2real
        file_name = input_map_b["file_name"]

        for i in range(0, len(file_name)):
            impath = withshadows_path + file_name[i] + ".png"
            torchvision.utils.save_image(img_b2a[i], impath, normalize = True)
            # print("Saved image (with shadows) : ", impath)

    #measures the performance of a given batch and stores it
    def measure_and_store(self, input_map):
        use_tanh = ConfigHolder.getInstance().get_network_attribute("use_tanh", False)
        img_a2b, img_b2a = self.img2img_t.test(input_map) #a2b --> real2synth, b2a --> synth2real
        file_name = input_map["file_name"]
        target = input_map["img_a"]

        if(use_tanh):
            img_b2a = tensor_utils.normalize_to_01(img_b2a)
            target = tensor_utils.normalize_to_01(target)

        psnr_result = kornia.metrics.psnr(img_b2a, target, torch.max(target).item())
        self.psnr_results.append(psnr_result.item())

        l1_result = self.l1_loss(img_b2a, target).cpu()
        self.l1_results.append(l1_result)

        mse_result = self.mse_loss(img_b2a, target).cpu()
        self.mse_results.append(mse_result)

    def visualize_results(self, input_map, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_st_version_name()
        self.img2img_t.visdom_visualize(input_map, "Test - " + version_name + " " + dataset_title)

    def report_metrics(self, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_st_version_name()

        psnr_mean = np.round(np.mean(self.psnr_results), 4)
        self.psnr_results.clear()

        l1_mean = np.round(np.float32(np.mean(self.l1_results)), 4) #ISSUE: ROUND to 4 sometimes cause inf
        self.l1_results.clear()

        mse_mean = np.round(np.mean(self.mse_results), 4)
        self.mse_results.clear()

        last_epoch = global_config.last_epoch_st
        self.visdom_reporter.plot_text(dataset_title + " Results - " + version_name + " Last epoch: " + str(last_epoch) + "<br>"
                                        + "Dataset: " + str(global_config.dataset_target) + "<br>"
                                       + "PSNR: " +str(psnr_mean) + "<br>" 
                                       "Abs Rel: " + str(l1_mean) + "<br>"
                                        "Sqr Rel: " + str(mse_mean) + "<br>")
