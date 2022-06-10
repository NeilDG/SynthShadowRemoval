# -*- coding: utf-8 -*-
# Cycle consistent relighting trainer
import random

import kornia
from model import iteration_table, embedding_network
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from model.modules import image_pool
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from model.iteration_table import IterationTable
from transforms import iid_transforms
from utils import plot_utils
from utils import tensor_utils
from custom_losses import ssim_loss
import lpips

class IIDTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

        self.iteration = opts.iteration
        self.it_table = iteration_table.IterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration, IterationTable.NetworkType.ALBEDO)
        self.da_enabled = opts.da_enabled

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.default_light_color = "255,255,255"
        # self.default_light_color = "236,193,178"
        # self.default_light_color = "88,100,109"

        self.D_A_pool = image_pool.ImagePool(50)
        self.D_S_pool = image_pool.ImagePool(50)

        self.iid_op = iid_transforms.IIDTransform()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if(self.da_enabled == 1):
            self.initialize_da_network(opts.da_version_name)
            self.initialize_shading_network(net_config, num_blocks, 6)
            self.initialize_albedo_network(net_config, num_blocks, 6)
        else:
            self.initialize_shading_network(net_config, num_blocks, 3)
            self.initialize_albedo_network(net_config, num_blocks, 3)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_S.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_S.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.optimizerG_albedo = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD_albedo = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_albedo, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_albedo, patience=100000 / self.batch_size, threshold=0.00005)

        self.initialize_dict()

        self.fp16_scaler_s = amp.GradScaler()  # for automatic mixed precision

    def initialize_da_network(self, da_version_name):
        self.embedder = embedding_network.EmbeddingNetworkFFA(blocks=6).to(self.gpu_device)
        checkpoint = torch.load("checkpoint/" + da_version_name + ".pt", map_location=self.gpu_device)
        self.embedder.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded embedding network: ", da_version_name)

        self.decoder_fixed = embedding_network.DecodingNetworkFFA().to(self.gpu_device)
        print("Loaded fixed decoder network")

    def initialize_shading_network(self, net_config, num_blocks, input_nc):
        if (net_config == 1):
            self.G_S = cycle_gan.Generator(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_S = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_S = cycle_gan.Generator(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
        elif (net_config == 4):
            self.G_S = unet_gan.UnetGeneratorV2(input_nc=input_nc, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_S = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_S = cycle_gan.Discriminator(input_nc=1).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_albedo_network(self, net_config, num_blocks, input_nc):
        if (net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = unet_gan.UnetGeneratorV2(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

    # def initialize_shadow_network(self, net_config, num_blocks, input_nc):
    #     if (net_config == 1):
    #         self.G_Z = cycle_gan.Generator(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
    #     elif (net_config == 2):
    #         self.G_Z = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
    #     elif (net_config == 3):
    #         self.G_Z = cycle_gan.Generator(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, use_cbam=True).to(self.gpu_device)
    #     elif (net_config == 4):
    #         self.G_Z = unet_gan.UnetGeneratorV2(input_nc=input_nc, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
    #     else:
    #         self.G_Z = cycle_gan.GeneratorV2(input_nc=input_nc, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)
    #
    #     self.D_Z = cycle_gan.Discriminator(input_nc=1).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_s[constants.SSIM_LOSS_KEY] = []
        self.GRADIENT_LOSS_KEY = "GRADIENT_LOSS_KEY"
        self.losses_dict_s[self.GRADIENT_LOSS_KEY ] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []
        # self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        # self.losses_dict[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shading + Shadow G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_s[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        # self.caption_dict[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"

        # what to store in visdom?
        self.losses_dict_a = {}
        self.losses_dict_a[constants.G_LOSS_KEY] = []
        self.losses_dict_a[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_a[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_a[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_a[constants.SSIM_LOSS_KEY] = []
        self.losses_dict_a[self.GRADIENT_LOSS_KEY] = []
        self.losses_dict_a[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_REAL_LOSS_KEY] = []
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict_a = {}
        self.caption_dict_a[constants.G_LOSS_KEY] = "Albedo G loss per iteration"
        self.caption_dict_a[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_a[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_a[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_a[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_a[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_a[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_a[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_a[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"

    def normalize(self, light_angle):
        std = light_angle / 360.0
        min = -1.0
        max = 1.0
        scaled = std * (max - min) + min

        return scaled

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.mse_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def gradient_loss_term(self, pred, target):
        pred_gradient = kornia.filters.spatial_gradient(pred)
        target_gradient = kornia.filters.spatial_gradient(target)

        return self.mse_loss(pred_gradient, target_gradient)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def ssim_loss(self, pred, target):
        pred_normalized = (pred * 0.5) + 0.5
        target_normalized = (target * 0.5) + 0.5

        return self.ssim_loss(pred_normalized, target_normalized)

    def update_penalties(self, adv_weight, rgb_l1_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.rgb_l1_weight = rgb_l1_weight

        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))

        print("======ALBEDO======")
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

        print("======SHADING======")
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADING)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

        print("======SHADOW======")
        print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("LPIP weight: ", str(self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("SSIM weight: ", str(self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)))
        print("Gradient weight: ", str(self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)))

    def train(self, input_rgb_tensor, albedo_tensor, shading_tensor):
        self.train_shading(input_rgb_tensor, shading_tensor)
        self.train_albedo(input_rgb_tensor, albedo_tensor)

    def reshape_input(self, input_tensor):
        rgb_embedding, w1, w2, w3 = self.embedder.get_embedding(input_tensor)
        rgb_feature_rep = self.decoder_fixed.get_decoding(input_tensor, rgb_embedding, w1, w2, w3)

        return torch.cat([input_tensor, rgb_feature_rep], 1)

    def train_shading(self, input_rgb_tensor, shading_tensor):
        with amp.autocast():
            if (self.da_enabled == 1):
                input_rgb_tensor = self.reshape_input(input_rgb_tensor)

            self.optimizerD_shading.zero_grad()

            #shading discriminator
            rgb2shading = self.G_S(input_rgb_tensor)
            self.D_S.train()
            prediction = self.D_S(shading_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_S_real_loss = self.adversarial_loss(self.D_S(shading_tensor), real_tensor) * self.adv_weight
            D_S_fake_loss = self.adversarial_loss(self.D_S_pool.query(self.D_S(rgb2shading.detach())), fake_tensor) * self.adv_weight

            errD = D_S_real_loss + D_S_fake_loss

            self.fp16_scaler_s.scale(errD).backward()
            if (self.fp16_scaler_s.scale(errD).item() > 0.0):
                self.fp16_scaler_s.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            #shading generator
            self.G_S.train()
            rgb2shading = self.G_S(input_rgb_tensor)
            S_likeness_loss = self.l1_loss(rgb2shading, shading_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_lpip_loss = self.lpip_loss(rgb2shading, shading_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_ssim_loss = self.ssim_loss(rgb2shading, shading_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_gradient_loss = self.gradient_loss_term(rgb2shading, shading_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADING)
            prediction = self.D_S(rgb2shading)
            real_tensor = torch.ones_like(prediction)
            S_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            # shadow generator
            # self.G_Z.train()
            # rgb2shadow = self.G_Z(input_rgb_tensor)
            # Z_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            # Z_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            # Z_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            # Z_gradient_loss = self.gradient_loss_term(rgb2shadow, shadow_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            # prediction = self.D_Z(rgb2shadow)
            # real_tensor = torch.ones_like(prediction)
            # Z_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = S_likeness_loss + S_lpip_loss + S_ssim_loss + S_gradient_loss + S_adv_loss

            self.fp16_scaler_s.scale(errG).backward()
            self.fp16_scaler_s.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler_s.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(S_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(S_lpip_loss.item())
            self.losses_dict_s[constants.SSIM_LOSS_KEY].append(S_ssim_loss.item())
            self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(S_gradient_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(S_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_S_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_S_real_loss.item())

    def train_albedo(self, input_rgb_tensor, albedo_tensor):
        with amp.autocast():
            self.G_S.eval()

            # produce initial albedo
            rgb2albedo = self.G_A(input_rgb_tensor)

            # albedo discriminator
            self.D_A.train()
            self.optimizerD_albedo.zero_grad()
            prediction = self.D_A(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A_pool.query(self.D_A(rgb2albedo.detach())), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler_s.scale(errD).backward()

            if (self.fp16_scaler_s.scale(errD).item() > 0.0):
                self.schedulerD_albedo.step(errD)
                self.fp16_scaler_s.step(self.optimizerD_albedo)

            self.optimizerG_albedo.zero_grad()

            # albedo generator
            self.G_A.train()
            rgb2albedo = self.G_A(input_rgb_tensor)
            A_likeness_loss = self.l1_loss(rgb2albedo, albedo_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_lpip_loss = self.lpip_loss(rgb2albedo, albedo_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_ssim_loss = self.ssim_loss(rgb2albedo, albedo_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_gradient_loss = self.gradient_loss_term(rgb2albedo, albedo_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            prediction = self.D_A(rgb2albedo)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = self.iid_op.produce_rgb(rgb2albedo, self.G_S(input_rgb_tensor))
            rgb_l1_loss = self.l1_loss(rgb_like, input_rgb_tensor) * self.rgb_l1_weight

            errG = A_likeness_loss + A_lpip_loss + A_ssim_loss + A_gradient_loss + A_adv_loss + rgb_l1_loss
            self.fp16_scaler_s.scale(errG).backward()
            self.schedulerG_albedo.step(errG)
            self.fp16_scaler_s.step(self.optimizerG_albedo)
            self.fp16_scaler_s.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_a[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_a[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_a[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
            self.losses_dict_a[constants.LPIP_LOSS_KEY].append(A_lpip_loss.item())
            self.losses_dict_a[constants.SSIM_LOSS_KEY].append(A_ssim_loss.item())
            self.losses_dict_a[self.GRADIENT_LOSS_KEY].append(A_gradient_loss.item())
            self.losses_dict_a[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict_a[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
            self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

    def test(self, input_rgb_tensor):
        with torch.no_grad():
            rgb2shading = self.G_S(input_rgb_tensor)
            rgb2albedo = self.G_A(input_rgb_tensor)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading)
        return rgb_like

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_s", iteration, self.losses_dict_s, self.caption_dict_s, constants.IID_CHECKPATH)
        # self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_a, self.caption_dict_a, constants.IID_CHECKPATH)

    def visdom_visualize(self, input_rgb_tensor, albedo_tensor, shading_tensor, label = "Train"):
        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)

                a, b, c, d = self.embedder.get_embedding(input_rgb_tensor)
                embedding_rep = self.decoder_fixed.get_decoding(input_rgb_tensor, a,b,c,d)
            else:
                input = input_rgb_tensor
                embedding_rep = input_rgb_tensor

            rgb2shading = self.G_S(input)
            rgb2albedo = self.G_A(input)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading)

            self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, str(label) + " RGB Reconstruction - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2albedo, str(label) + " RGB2Albedo images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Albedo images - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shading, str(label) + " RGB2Shading images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_measure(self, input_rgb_tensor, albedo_tensor, shading_tensor, label="Training"):
        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)
            else:
                input = input_rgb_tensor

            rgb2shading = self.G_S(input)
            rgb2albedo = self.G_A(input)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading)

            # plot metrics
            # rgb2albedo = (rgb2albedo * 0.5) + 0.5
            albedo_tensor = (albedo_tensor * 0.5) + 0.5
            rgb2shading = (rgb2shading * 0.5) + 0.5
            shading_tensor = (shading_tensor * 0.5) + 0.5

            psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
            ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
            psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
            ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
            psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, input_rgb_tensor, max_val=1.0).item(), 4)
            ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, input_rgb_tensor, 5).item(), 4)
            display_text = str(label) + " - Versions: " + constants.IID_VERSION + constants.ITERATION + \
                           "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) + \
                           "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
                           "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)

            self.visdom_reporter.plot_text(display_text)

    # must have a shading generator network first
    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(rw_tensor)

                a, b, c, d = self.embedder.get_embedding(rw_tensor)
                embedding_rep = self.decoder_fixed.get_decoding(rw_tensor, a, b, c, d)
            else:
                input = rw_tensor
                embedding_rep = rw_tensor

            rgb2shading = self.G_S(input)
            rgb2albedo = self.G_A(input)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading)

            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(embedding_rep, "Real World Embeddings - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "Real World A2B images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_measure_gta(self, gta_rgb, gta_albedo):
        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(gta_rgb)
            else:
                input = gta_rgb

            rgb2shading = self.G_S(input)
            rgb2albedo = self.G_A(input)
            rgb_like = self.iid_op.produce_rgb(rgb2albedo, rgb2shading)

            self.visdom_reporter.plot_image(gta_albedo, "GTA Albedo - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb2albedo, "GTA Albedo-Like - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shading, "GTA Shading-Like - " + constants.IID_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(gta_rgb, "GTA RGB - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "GTA RGB-Like - " + constants.IID_VERSION + constants.ITERATION)

    def infer_albedo(self, rw_tensor):
        with torch.no_grad():
            if (self.da_enabled == 1):
                input = self.reshape_input(rw_tensor)
            else:
                input = rw_tensor
            return self.G_A(input)

    def infer_shading(self, rw_tensor):
        with torch.no_grad():
            if (self.da_enabled == 1):
                rw_tensor = self.reshape_input(rw_tensor)

            return self.G_S(rw_tensor)

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_S.load_state_dict(checkpoint[constants.GENERATOR_KEY + "S"])
        self.D_S.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "S"])

        self.optimizerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"])
        self.optimizerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
        self.schedulerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "A"])
        self.schedulerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])

        self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"])
        self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"])
        self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "S"])
        self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "S"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}

        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        optimizerGalbedo_state_dict = self.optimizerG_albedo.state_dict()
        optimizerDalbedo_state_dict = self.optimizerD_albedo.state_dict()

        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()
        schedulerGalbedo_state_dict = self.schedulerG_albedo.state_dict()
        schedulerDalbedo_state_dict = self.schedulerD_albedo.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDalbedo_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "A"] = schedulerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDalbedo_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "S"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "S"] = schedulerDshading_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}

        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        optimizerGalbedo_state_dict = self.optimizerG_albedo.state_dict()
        optimizerDalbedo_state_dict = self.optimizerD_albedo.state_dict()

        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()
        schedulerGalbedo_state_dict = self.schedulerG_albedo.state_dict()
        schedulerDalbedo_state_dict = self.schedulerD_albedo.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDalbedo_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "A"] = schedulerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDalbedo_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "S"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "S"] = schedulerDshading_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))