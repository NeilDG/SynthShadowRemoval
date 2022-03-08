# -*- coding: utf-8 -*-
# Cycle consistent relighting trainer
import kornia

from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from utils import plot_utils
from utils import tensor_utils
from custom_losses import ssim_loss
import lpips

class RelightingTrainer:

    def __init__(self, gpu_device, opts, use_bce):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = use_bce

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.default_light_color = "255,255,255"

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        self.initialize_albedo_network(net_config, num_blocks)
        self.initialize_shading_network(net_config, num_blocks)
        self.initialize_shadow_network(net_config, num_blocks)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_S.parameters(), self.G_Z.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_S.parameters(), self.D_Z.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_albedo_network(self, net_config, num_blocks):
        if (net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = unet_gan.UnetGeneratorV2(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=3, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_shading_network(self, net_config, num_blocks):
        if (net_config == 1):
            self.G_S = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_S = unet_gan.UnetGenerator(input_nc=3, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_S = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 4):
            self.G_S = unet_gan.UnetGeneratorV2(input_nc=3, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_S = cycle_gan.GeneratorV2(input_nc=3, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_S = cycle_gan.Discriminator(input_nc=1, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_shadow_network(self, net_config, num_blocks):
        if (net_config == 1):
            self.G_Z = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_Z = unet_gan.UnetGenerator(input_nc=3, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_Z = cycle_gan.Generator(input_nc=3, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 4):
            self.G_Z = unet_gan.UnetGeneratorV2(input_nc=3, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_Z = cycle_gan.GeneratorV2(input_nc=3, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_Z = cycle_gan.Discriminator(input_nc=1, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.LPIP_LOSS_KEY] = []
        # self.losses_dict[constants.SSIM_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.losses_dict[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        # self.caption_dict[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"

    def normalize(self, light_angle):
        std = light_angle / 360.0
        min = -1.0
        max = 1.0
        scaled = std * (max - min) + min

        return scaled

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.l1_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def l1_loss(self, pred, target):
        return self.l1_loss(pred, target)

    def bce_loss_term(self, pred, target):
        return self.bce_loss(pred, target)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def ssim_loss(self, pred, target):
        return kornia.losses.ssim_loss(pred, target)

    def update_penalties(self, adv_weight, l1_weight, lpip_weight, ssim_weight, bce_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.lpip_weight = lpip_weight
        self.ssim_weight = ssim_weight
        self.bce_weight = bce_weight

        print("Version: ", constants.RELIGHTING_CHECKPATH)
        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))
        print("Likeness weight: ", str(self.l1_weight))
        print("LPIP weight: ", str(self.lpip_weight))
        print("SSIM weight: ", str(self.ssim_weight))
        print("BCE weight: ", str(self.bce_weight))

    # def produce_rgb(self, albedo_tensor, shading_tensor, light_color, shadowmap_tensor):
    #     albedo_tensor = albedo_tensor.transpose(0, 1)
    #     shading_tensor = shading_tensor.transpose(0, 1)
    #     shadowmap_tensor = shadowmap_tensor.transpose(0, 1)
    #     light_color = torch.from_numpy(np.asarray(light_color.split(","), dtype=np.int32))
    #
    #     # print("Shading Range: ", torch.min(shading_tensor).item(), torch.max(shading_tensor).item(), " Mean: ", torch.mean(shading_tensor).item())
    #     # print("ShadowMap Range: ", torch.min(shadowmap_tensor).item(), torch.max(shadowmap_tensor).item(), " Mean: ", torch.mean(shading_tensor).item())
    #     # print("Light Range: ", light_color)
    #
    #     # normalize/remove normalization
    #     albedo_tensor = (albedo_tensor * 0.5) + 0.5
    #     shading_tensor = (shading_tensor * 0.5) + 0.5
    #     shadowmap_tensor = (shadowmap_tensor * 0.5) + 0.5
    #     light_color = light_color / 255.0
    #
    #     rgb_img_like = torch.full_like(albedo_tensor, 0)
    #     rgb_img_like[0] = torch.clip(albedo_tensor[0] * shading_tensor[0] * light_color[0] * shadowmap_tensor, 0.0, 1.0)
    #     rgb_img_like[1] = torch.clip(albedo_tensor[1] * shading_tensor[1] * light_color[1] * shadowmap_tensor, 0.0, 1.0)
    #     rgb_img_like[2] = torch.clip(albedo_tensor[2] * shading_tensor[2] * light_color[2] * shadowmap_tensor, 0.0, 1.0)
    #
    #     rgb_img_like = rgb_img_like.transpose(0, 1)
    #     return rgb_img_like

    def train(self, input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor, target_rgb_tensor):
        with amp.autocast():
            self.optimizerD.zero_grad()

            #albedo discriminator
            rgb2albedo = self.G_A(input_rgb_tensor)
            self.D_A.train()
            prediction = self.D_A(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_A_real_loss = self.adversarial_loss(self.D_A(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(rgb2albedo.detach()), fake_tensor) * self.adv_weight

            #shading discriminator
            rgb2shading = self.G_S(input_rgb_tensor)
            self.D_S.train()
            prediction = self.D_S(shading_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_S_real_loss = self.adversarial_loss(self.D_S(shading_tensor), real_tensor) * self.adv_weight
            D_S_fake_loss = self.adversarial_loss(self.D_S(rgb2shading.detach()), fake_tensor) * self.adv_weight

            # shadow discriminator
            rgb2shadow = self.G_Z(input_rgb_tensor)
            self.D_Z.train()
            prediction = self.D_Z(shadow_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_Z_real_loss = self.adversarial_loss(self.D_Z(shadow_tensor), real_tensor) * self.adv_weight
            D_Z_fake_loss = self.adversarial_loss(self.D_Z(rgb2shadow.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss + D_S_real_loss + D_S_fake_loss + D_Z_real_loss + D_Z_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.optimizerG.zero_grad()

            #albedo generator
            self.G_A.train()
            rgb2albedo = self.G_A(input_rgb_tensor)
            A_likeness_loss = self.l1_loss(rgb2albedo, albedo_tensor) * self.l1_weight
            A_lpip_loss = self.lpip_loss(rgb2albedo, albedo_tensor) * self.lpip_weight
            # A_ssim_loss = self.ssim_loss(rgb2albedo, albedo_tensor) * self.ssim_weight
            A_bce_loss = self.bce_loss_term(rgb2albedo, albedo_tensor) * self.bce_weight
            prediction = self.D_A(rgb2albedo)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            #shading generator
            self.G_S.train()
            rgb2shading = self.G_S(input_rgb_tensor)
            S_likeness_loss = self.l1_loss(rgb2shading, shading_tensor) * self.l1_weight
            S_lpip_loss = self.lpip_loss(rgb2shading, shading_tensor) * self.lpip_weight
            # S_ssim_loss = self.ssim_loss(rgb2shading, shading_tensor) * self.ssim_weight
            S_bce_loss = self.bce_loss_term(rgb2shading, shading_tensor) * self.bce_weight
            prediction = self.D_S(rgb2shading)
            real_tensor = torch.ones_like(prediction)
            S_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            # shading generator
            self.G_Z.train()
            rgb2shadow = self.G_Z(input_rgb_tensor)
            Z_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.l1_weight
            Z_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.lpip_weight
            # Z_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.ssim_weight
            Z_bce_loss = self.bce_loss_term(rgb2shadow, shadow_tensor) * self.bce_weight
            prediction = self.D_Z(rgb2shadow)
            real_tensor = torch.ones_like(prediction)
            Z_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = tensor_utils.produce_rgb(rgb2albedo, rgb2shading, self.default_light_color, rgb2shadow)
            rgb_l1_loss = self.l1_loss(rgb_like, target_rgb_tensor) * self.l1_weight

            errG = A_likeness_loss + A_lpip_loss + A_bce_loss + A_adv_loss + \
                   S_likeness_loss + S_lpip_loss + S_bce_loss + S_adv_loss + \
                   Z_likeness_loss + Z_lpip_loss + Z_bce_loss + Z_adv_loss + rgb_l1_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item() + S_likeness_loss.item() + Z_likeness_loss.item())
            self.losses_dict[constants.LPIP_LOSS_KEY].append(A_lpip_loss.item() + S_lpip_loss.item() + Z_lpip_loss.item())
            # self.losses_dict[constants.SSIM_LOSS_KEY].append(A_ssim_loss.item() + S_ssim_loss.item() + Z_ssim_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + S_adv_loss.item() + Z_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item() + D_S_fake_loss.item() + D_Z_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item() + D_S_real_loss.item() + D_Z_real_loss.item())
            self.losses_dict[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

    def test(self, input_rgb_tensor):
        with torch.no_grad():
            rgb2albedo = self.G_A(input_rgb_tensor)
            rgb2shading = self.G_S(input_rgb_tensor)
            rgb2shadow = self.G_Z(input_rgb_tensor)
            rgb_like = tensor_utils.produce_rgb(rgb2albedo, rgb2shading, self.default_light_color, rgb2shadow)
        return rgb_like

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.RELIGHTING_CHECKPATH)

    def visdom_visualize(self, input_rgb_tensor, albedo_tensor, shading_tensor, shadow_tensor, target_rgb_tensor, label = "Training"):
        with torch.no_grad():
            rgb2albedo = self.G_A(input_rgb_tensor)
            rgb2shading = self.G_S(input_rgb_tensor)
            rgb2shadow = self.G_Z(input_rgb_tensor)
            rgb_like = tensor_utils.produce_rgb(rgb2albedo, rgb2shading, self.default_light_color, rgb2shadow)

            self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, str(label) + " RGB Reconstruction - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(target_rgb_tensor, str(label) + " Target RGB Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2albedo, str(label) + " RGB2Albedo images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Albedo images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shading, str(label) + " RGB2Shading images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(rgb2shadow, str(label) + " RGB2Shadow images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

            # plot metrics
            # rgb2albedo = (rgb2albedo * 0.5) + 0.5
            # albedo_tensor = (albedo_tensor * 0.5) + 0.5
            # rgb2shading = (rgb2shading * 0.5) + 0.5
            # shading_tensor = (shading_tensor * 0.5) + 0.5
            # rgb2shadow = (rgb2shadow * 0.5) + 0.5
            # shadow_tensor = (shadow_tensor * 0.5) + 0.5
            # target_rgb_tensor = (target_rgb_tensor * 0.5) + 0.5
            #
            # psnr_albedo = np.round(kornia.metrics.psnr(rgb2albedo, albedo_tensor, max_val=1.0).item(), 4)
            # ssim_albedo = np.round(1.0 - kornia.losses.ssim_loss(rgb2albedo, albedo_tensor, 5).item(), 4)
            # psnr_shading = np.round(kornia.metrics.psnr(rgb2shading, shading_tensor, max_val=1.0).item(), 4)
            # ssim_shading = np.round(1.0 - kornia.losses.ssim_loss(rgb2shading, shading_tensor, 5).item(), 4)
            # psnr_shadow = np.round(kornia.metrics.psnr(rgb2shadow, shadow_tensor, max_val=1.0).item(), 4)
            # ssim_shadow = np.round(1.0 - kornia.losses.ssim_loss(rgb2shadow, shadow_tensor, 5).item(), 4)
            # psnr_rgb = np.round(kornia.metrics.psnr(rgb_like, target_rgb_tensor, max_val=1.0).item(), 4)
            # ssim_rgb = np.round(1.0 - kornia.losses.ssim_loss(rgb_like, target_rgb_tensor, 5).item(), 4)
            # display_text = str(label) + " - Versions: " + constants.RELIGHTING_VERSION + constants.ITERATION +\
            #                "<br> Albedo PSNR: " + str(psnr_albedo) + "<br> Albedo SSIM: " + str(ssim_albedo) +\
            #                "<br> Shading PSNR: " + str(psnr_shading) + "<br> Shading SSIM: " + str(ssim_shading) + \
            #                "<br> Shadow PSNR: " + str(psnr_shadow) + "<br> Shadow SSIM: " + str(ssim_shadow) + \
            #                "<br> RGB Reconstruction PSNR: " + str(psnr_rgb) + "<br> RGB Reconstruction SSIM: " + str(ssim_rgb)
            #
            # self.visdom_reporter.plot_text(display_text)

    # must have a shading generator network first
    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rgb2albedo = self.G_A(rw_tensor)
            rgb2shading = self.G_S(rw_tensor)
            rgb2shadow = self.G_Z(rw_tensor)
            rgb_like = tensor_utils.produce_rgb(rgb2albedo, rgb2shading, self.default_light_color, rgb2shadow)

            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "Real World A2B images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_S.load_state_dict(checkpoint[constants.GENERATOR_KEY + "S"])
        self.D_S.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "S"])
        self.G_Z.load_state_dict(checkpoint[constants.GENERATOR_KEY + "Z"])
        self.D_Z.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "Z"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()

        netGZ_state_dict = self.G_Z.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict
        save_dict[constants.GENERATOR_KEY + "Z"] = netGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "Z"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.RELIGHTING_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()

        netGZ_state_dict = self.G_Z.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict
        save_dict[constants.GENERATOR_KEY + "Z"] = netGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "Z"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.RELIGHTING_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))