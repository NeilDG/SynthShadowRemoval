# -*- coding: utf-8 -*-
# Shading trainer used for training.
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

class ShadowRelightTrainer:

    def __init__(self, gpu_device, opts, use_bce):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = use_bce

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if (net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=2, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=2, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.Generator(input_nc=2, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=1, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.LPIP_LOSS_KEY] = []
        self.losses_dict[constants.SSIM_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"

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

        print("Version: ", constants.SHADOWMAP_RELIGHT_CHECKPATH)
        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))
        print("Likeness weight: ", str(self.l1_weight))
        print("LPIP weight: ", str(self.lpip_weight))
        print("SSIM weight: ", str(self.ssim_weight))
        print("BCE weight: ", str(self.bce_weight))

    def prepare_input(self, a_tensor, light_angle):
        # light_angle = self.normalize(light_angle)
        # light_angle_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], light_angle), 1)
        # print("Shapes: ", np.shape(a_tensor), np.shape(light_angle))
        concat_input = torch.cat([a_tensor, light_angle], 1)
        return concat_input

    def train(self, a_tensor, b_tensor, light_angle):
        with amp.autocast():
            a2b = self.G_A(self.prepare_input(a_tensor, light_angle))

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(b_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            # print("A2B shape: ", np.shape(a2b))
            D_A_real_loss = self.adversarial_loss(self.D_A(b_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(a2b.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            a2b = self.G_A(self.prepare_input(a_tensor, light_angle))

            likeness_loss = self.l1_loss(a2b, b_tensor) * self.l1_weight
            lpip_loss = self.lpip_loss(a2b, b_tensor) * self.lpip_weight
            ssim_loss = self.ssim_loss(a2b, b_tensor) * self.ssim_weight
            bce_loss_val = self.bce_loss_term(a2b, b_tensor) * self.bce_weight

            prediction = self.D_A(a2b)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_adv_loss + likeness_loss + lpip_loss + ssim_loss + bce_loss_val

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[constants.LPIP_LOSS_KEY].append(lpip_loss.item())
            self.losses_dict[constants.SSIM_LOSS_KEY].append(ssim_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def test(self, a_tensor, light_angle):
        with torch.no_grad():
            a2b = self.G_A(self.prepare_input(a_tensor, light_angle))
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH)

    def visdom_visualize(self, a_tensor, b_tensor, light_angle, a_test, b_test, light_angle_test):
        with torch.no_grad():
            a2b = self.G_A(self.prepare_input(a_tensor, light_angle))
            test_a2b = self.G_A(self.prepare_input(a_test, light_angle_test))

            self.visdom_reporter.plot_image(a_tensor, "Training A images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, "Training A2B images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_tensor, "B images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(a_test, "Test A images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_test, "Test B images - " + constants.SHADOWMAP_RELIGHT_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

class ShadowCustomGeneratorV1(cycle_gan.Generator):
    def __init__(self, input_nc=3, output_nc=3, downsampling_blocks = 2, n_residual_blocks=6, has_dropout = True):
        cycle_gan.Generator.__init__(self, input_nc, output_nc, downsampling_blocks, n_residual_blocks, has_dropout)

    def set_shadow_input(self, input_shadow_tensor):
        self.input_shadow_tensor = input_shadow_tensor

    def forward(self, x):
        return super().forward(x) * self.input_shadow_tensor

class ShadowCustomGeneratorV2(unet_gan.UnetGenerator):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        unet_gan.UnetGenerator.__init__(self, input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout, gpu_ids)

    def set_shadow_input(self, input_shadow_tensor):
        self.input_shadow_tensor = input_shadow_tensor

    def forward(self, x):
        return super().forward(x) * self.input_shadow_tensor

class ShadowRelightTrainerRGB:

    def __init__(self, gpu_device, opts, use_bce):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = use_bce
        self.default_light_color = "255,255,255"

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if (net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=5, output_nc=1, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=5, output_nc=1, num_downs=num_blocks).to(self.gpu_device)
        elif(net_config == 3):
            self.G_A = cycle_gan.Generator(input_nc=5, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = ShadowCustomGeneratorV1(input_nc=5, output_nc=1, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        else:
            self.G_A = ShadowCustomGeneratorV2(input_nc=5, output_nc=1, num_downs=num_blocks).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=1, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.LPIP_LOSS_KEY] = []
        self.losses_dict[constants.SSIM_LOSS_KEY] = []
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
        self.caption_dict[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"
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

        print("Version: ", constants.SHADOWMAP_RELIGHT_CHECKPATH)
        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))
        print("Likeness weight: ", str(self.l1_weight))
        print("LPIP weight: ", str(self.lpip_weight))
        print("SSIM weight: ", str(self.ssim_weight))
        print("BCE weight: ", str(self.bce_weight))

    def prepare_input(self, input_shadow_tensor, input_rgb_tensor, light_angle):
        # light_angle = self.normalize(light_angle)
        # light_angle_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], light_angle), 1)
        # print("Shapes: ", np.shape(a_tensor), np.shape(light_angle))
        concat_input = torch.cat([input_shadow_tensor, input_rgb_tensor, light_angle], 1)
        return concat_input

    def train(self, input_rgb_tensor, input_shadow_tensor, light_angle_tensor,
              albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor):
        with amp.autocast():
            self.G_A.set_shadow_input(input_shadow_tensor)
            shadow_like = self.G_A(self.prepare_input(input_shadow_tensor, input_rgb_tensor, light_angle_tensor))

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(target_shadow_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            # print("A2B shape: ", np.shape(a2b))
            D_A_real_loss = self.adversarial_loss(self.D_A(target_shadow_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(shadow_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            self.G_A.set_shadow_input(input_shadow_tensor)
            shadow_like = self.G_A(self.prepare_input(input_shadow_tensor, input_rgb_tensor, light_angle_tensor))

            likeness_loss = self.l1_loss(shadow_like, target_shadow_tensor) * self.l1_weight
            lpip_loss = self.lpip_loss(shadow_like, target_shadow_tensor) * self.lpip_weight
            ssim_loss = self.ssim_loss(shadow_like, target_shadow_tensor) * self.ssim_weight
            bce_loss_val = self.bce_loss_term(shadow_like, target_shadow_tensor) * self.bce_weight

            prediction = self.D_A(shadow_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = tensor_utils.produce_rgb(albedo_tensor, shading_tensor, self.default_light_color, shadow_like)
            rgb_l1_loss = self.l1_loss(rgb_like, target_rgb_tensor) * self.l1_weight

            errG = A_adv_loss + likeness_loss + lpip_loss + ssim_loss + bce_loss_val + rgb_l1_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[constants.LPIP_LOSS_KEY].append(lpip_loss.item())
            self.losses_dict[constants.SSIM_LOSS_KEY].append(ssim_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
            self.losses_dict[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

    def test(self, input_shadow_tensor, input_rgb_tensor, light_angle):
        with torch.no_grad():
            self.G_A.set_shadow_input(input_shadow_tensor)
            a2b = self.G_A(self.prepare_input(input_shadow_tensor, input_rgb_tensor, light_angle))
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH)

    def visdom_visualize(self, input_rgb_tensor, input_shadow_tensor, light_angle_tensor,
              albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor, label="Training"):
        with torch.no_grad():
            self.G_A.set_shadow_input(input_shadow_tensor)
            shadow_like = self.G_A(self.prepare_input(input_shadow_tensor, input_rgb_tensor, light_angle_tensor))
            rgb_like = tensor_utils.produce_rgb(albedo_tensor, shading_tensor, self.default_light_color, shadow_like)

            self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, str(label) + " RGB Reconstruction - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(target_rgb_tensor, str(label) + " Target RGB Images - " + constants.RELIGHTING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(shadow_like, str(label) + " RGB2Shadow images - " + constants.RELIGHTING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(target_shadow_tensor, str(label) + " Shadow images - " + constants.RELIGHTING_VERSION + constants.ITERATION)



    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.SHADOWMAP_RELIGHT_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))