# -*- coding: utf-8 -*-
# Render trainer used for training.
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

class RenderMapsTrainer:

    def __init__(self, gpu_device,  opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce
        self.use_mask = opts.use_mask

        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        net_config = opts.net_config

        if(net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif(net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            self.G_A = ffa.FFA(gps=3, blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 4):
            self.G_A = cycle_gan.GeneratorV3(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 5):
            self.G_A = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=True).to(self.gpu_device)
        else:
            self.G_A = cycle_gan.GeneratorV2(input_nc=3, output_nc=3, n_residual_blocks=num_blocks, has_dropout=False, multiply=False).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator

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
        HYPERPARAMS_PATH = "checkpoint/" + constants.MAPPER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.MAPPER_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.l1_weight), file=f)
            print("LPIP weight: ", str(self.lpip_weight), file=f)
            print("SSIM weight: ", str(self.ssim_weight), file=f)
            print("BCE weight: ", str(self.bce_weight), file=f)

    def train(self, a_tensor, b_tensor, train_mask):
        with amp.autocast():
            a2b = self.G_A(a_tensor)
            if (self.use_mask == 1):
                a2b = torch.mul(a2b, train_mask)

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

            a2b = self.G_A(a_tensor)
            if (self.use_mask == 1):
                a2b = torch.mul(a2b, train_mask)

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

    def test(self, a_tensor, train_mask):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
            if (self.use_mask == 1):
                a2b = torch.mul(a2b, train_mask)
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.MAPPER_CHECKPATH)

    def visdom_visualize(self, a_tensor, b_tensor, a_test, b_test):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
            test_a2b = self.G_A(a_test)
            a_tensor = (a_tensor * 0.5) + 0.5
            a_test = (a_test * 0.5) + 0.5

            self.visdom_reporter.plot_image(a_tensor, "Training A images - " + constants.MAPPER_VERSION + constants.ITERATION, False)
            self.visdom_reporter.plot_image(a2b, "Training A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_tensor, "B images - " + constants.MAPPER_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(a_test, "Test A images - " + constants.MAPPER_VERSION + constants.ITERATION, False)
            self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_test, "Test B images - " + constants.MAPPER_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rw2b = self.G_A(rw_tensor)
            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)


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

        torch.save(save_dict, constants.MAPPER_CHECKPATH + ".checkpt")
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

        torch.save(save_dict, constants.MAPPER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))