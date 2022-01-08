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
from custom_losses import ssim_loss
import lpips

class ShadingTrainer:

    def __init__(self, gpu_device,  opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce
        self.light_angle = opts.light_angle
        self.light_angle = self.normalize(self.light_angle)

        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if(net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=4, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif(net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=4, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = ffa.FFA(gps=4, blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = cycle_gan.Generator(input_nc=4, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 5):
            self.G_A = cycle_gan.GeneratorV2(input_nc=4, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=True).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=4, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=3, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

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

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.SHADING_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.SHADING_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.l1_weight), file=f)
            print("LPIP weight: ", str(self.lpip_weight), file=f)
            print("SSIM weight: ", str(self.ssim_weight), file=f)
            print("BCE weight: ", str(self.bce_weight), file=f)

    def train(self, a_tensor, b_tensor):
        with amp.autocast():
            light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([a_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(b_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(b_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(a2b.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([a_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

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

    def test(self, a_tensor):
        with torch.no_grad():
            light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([a_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.SHADING_CHECKPATH)

    def visdom_visualize(self, a_tensor, b_tensor, a_test, b_test):
        with torch.no_grad():
            light_tensor = torch.unsqueeze(torch.full_like(a_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([a_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

            test_light_tensor = torch.unsqueeze(torch.full_like(a_test[:, 0, :, :], self.light_angle), 1)
            a_test_input = torch.cat([a_test, test_light_tensor], 1)
            test_a2b = self.G_A(a_test_input)

            self.visdom_reporter.plot_image(a_tensor, "Training A images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, "Training A2B images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_tensor, "B images - " + constants.SHADING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(a_test, "Test A images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_test, "Test B images - " + constants.SHADING_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rw_light_tensor = torch.unsqueeze(torch.full_like(rw_tensor[:, 0, :, :], self.light_angle), 1)
            rw_input = torch.cat([rw_tensor, rw_light_tensor], 1)
            rw2b = self.G_A(rw_input)

            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.SHADING_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
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

        torch.save(save_dict, constants.SHADING_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
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

        torch.save(save_dict, constants.SHADING_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

class ShadingTrainerAlbedo:

    def __init__(self, gpu_device,  opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce
        self.light_angle = opts.light_angle
        self.light_angle = self.normalize(self.light_angle)

        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if(net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif(net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=7, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = ffa.FFA(gps=7, blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = cycle_gan.Generator(input_nc=7, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        elif (net_config == 5):
            self.G_A = cycle_gan.GeneratorV2(input_nc=7, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=True).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=7, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(input_nc=3, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

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

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.SHADING_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.SHADING_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.l1_weight), file=f)
            print("LPIP weight: ", str(self.lpip_weight), file=f)
            print("SSIM weight: ", str(self.ssim_weight), file=f)
            print("BCE weight: ", str(self.bce_weight), file=f)

    def train(self, input_tensor, albedo_tensor, result_tensor):
        with amp.autocast():
            light_tensor = torch.unsqueeze(torch.full_like(input_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([input_tensor, albedo_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(result_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(result_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(a2b.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            light_tensor = torch.unsqueeze(torch.full_like(input_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([input_tensor, albedo_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

            likeness_loss = self.l1_loss(a2b, result_tensor) * self.l1_weight
            lpip_loss = self.lpip_loss(a2b, result_tensor) * self.lpip_weight
            ssim_loss = self.ssim_loss(a2b, result_tensor) * self.ssim_weight
            bce_loss_val = self.bce_loss_term(a2b, result_tensor) * self.bce_weight

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

    def test(self, input_tensor, albedo_tensor):
        with torch.no_grad():
            light_tensor = torch.unsqueeze(torch.full_like(input_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([input_tensor, albedo_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.SHADING_CHECKPATH)

    def visdom_visualize(self, input_tensor, albedo_tensor, result_tensor, input_test, albedo_test, result_test):
        with torch.no_grad():
            light_tensor = torch.unsqueeze(torch.full_like(input_tensor[:, 0, :, :], self.light_angle), 1)
            a_input = torch.cat([input_tensor, albedo_tensor, light_tensor], 1)
            a2b = self.G_A(a_input)

            test_light_tensor = torch.unsqueeze(torch.full_like(input_test[:, 0, :, :], self.light_angle), 1)
            a_test_input = torch.cat([input_test, albedo_test, test_light_tensor], 1)
            test_a2b = self.G_A(a_test_input)

            self.visdom_reporter.plot_image(input_tensor, "Training A images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_tensor, "Training Albedo images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, "Training A2B images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(result_tensor, "B images - " + constants.SHADING_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(input_test, "Test A images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_test, "Test Albedo images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + constants.SHADING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(result_test, "Test B images - " + constants.SHADING_VERSION + constants.ITERATION)

    # def visdom_infer(self, rw_tensor):
    #     with torch.no_grad():
    #         rw_light_tensor = torch.unsqueeze(torch.full_like(rw_tensor[:, 0, :, :], self.light_angle), 1)
    #         rw_input = torch.cat([rw_tensor, rw_light_tensor], 1)
    #         rw2b = self.G_A(rw_input)
    #
    #         self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.SHADING_VERSION + constants.ITERATION)
    #         self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.SHADING_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
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

        torch.save(save_dict, constants.SHADING_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
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

        torch.save(save_dict, constants.SHADING_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))