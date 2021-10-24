# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import itertools
import os
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.utils as vutils
import constants
from model import vanilla_cycle_gan as cycle_gan
from model import ffa_gan
from model import unet_gan
from utils import plot_utils


class TransferTrainer:

    def __init__(self, gpu_device, batch_size, g_lr, d_lr, num_blocks):
        self.gpu_device = gpu_device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.G_A = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        self.G_B = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator

        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_B = cycle_gan.Discriminator().to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[constants.SMOOTHNESS_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_B_REAL_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.IDENTITY_LOSS_KEY] = "Identity loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.SMOOTHNESS_LOSS_KEY] = "Smoothness loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"

    def update_penalties(self, adv_weight, id_weight, likeness_weight, cycle_weight, smoothness_weight, comments):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.id_weight = id_weight
        self.likeness_weight = likeness_weight
        self.cycle_weight = cycle_weight
        self.smoothness_weight = smoothness_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.STYLE_TRANSFER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.STYLE_TRANSFER_CHECKPATH, file=f)
            print("Comment: ", comments, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Identity weight: ", str(self.id_weight), file=f)
            print("Likeness weight: ", str(self.likeness_weight), file=f)
            print("Smoothness weight: ", str(self.smoothness_weight), file=f)
            print("Cycle weight: ", str(self.cycle_weight), file=f)
            print("====================================", file=f)

    def adversarial_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        # loss = nn.BCEWithLogitsLoss()
        # return loss(pred, target)

    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def smoothness_loss(self, pred, target):
        loss = nn.L1Loss()
        pred_blur = kornia.gaussian_blur2d(pred, (7, 7), (5.5, 5.5))
        target_blur = kornia.gaussian_blur2d(target, (7, 7), (5.5, 5.5))

        return loss(pred_blur, target_blur)

    def train(self, dirty_tensor, clean_tensor):
        with amp.autocast():
            clean_like = self.G_A(dirty_tensor)
            dirty_like = self.G_B(clean_tensor)

            self.D_A.train()
            self.D_B.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(clean_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(clean_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight

            prediction = self.D_B(dirty_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.adversarial_loss(self.D_B(dirty_tensor), real_tensor) * self.adv_weight
            D_B_fake_loss = self.adversarial_loss(self.D_B(dirty_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
            # errD.backward()
            # self.optimizerD.step()
            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.G_B.train()
            self.optimizerG.zero_grad()

            identity_like = self.G_A(clean_tensor)
            clean_like = self.G_A(dirty_tensor)

            A_identity_loss = self.identity_loss(identity_like, clean_tensor) * self.id_weight
            A_likeness_loss = self.likeness_loss(clean_like, clean_tensor) * self.likeness_weight
            A_smoothness_loss = self.smoothness_loss(clean_like, clean_tensor) * self.smoothness_weight
            A_cycle_loss = self.cycle_loss(self.G_B(self.G_A(dirty_tensor)), dirty_tensor) * self.cycle_weight

            identity_like = self.G_B(dirty_tensor)
            dirty_like = self.G_B(clean_tensor)
            B_identity_loss = self.identity_loss(identity_like, dirty_tensor) * self.id_weight
            B_likeness_loss = self.likeness_loss(dirty_like, dirty_tensor) * self.likeness_weight
            B_smoothness_loss = self.smoothness_loss(dirty_like, dirty_tensor) * self.smoothness_weight
            B_cycle_loss = self.cycle_loss(self.G_A(self.G_B(clean_tensor)), clean_tensor) * self.cycle_weight

            prediction = self.D_A(clean_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            prediction = self.D_B(dirty_like)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_identity_loss + B_identity_loss + A_likeness_loss + B_likeness_loss + A_smoothness_loss + B_smoothness_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(A_identity_loss.item() + B_identity_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
        self.losses_dict[constants.SMOOTHNESS_LOSS_KEY].append(A_smoothness_loss.item() + B_smoothness_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())

    def test(self, styled_tensor):
        with torch.no_grad():
            albedo_like = self.G_A(styled_tensor)
            return albedo_like

    def visdom_report(self, iteration, a_tensor, b_tensor, test_a_tensor, test_b_tensor):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
            test_a2b = self.G_A(test_a_tensor)

            b2a = self.G_B(b_tensor)
            test_b2a = self.G_B(test_b_tensor)

        # report to visdom
        self.visdom_reporter.plot_finegrain_loss(str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION), iteration, self.losses_dict, self.caption_dict)
        self.visdom_reporter.plot_image(a_tensor, "Training A images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(a2b, "Training A2B images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(b_tensor, "Training B images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(b2a, "Training B2A images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_a_tensor, "Test A images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_b_tensor, "Test B images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))
        self.visdom_reporter.plot_image(test_b2a, "Test B2A images - " + str(constants.STYLE_TRANSFER_VERSION) + str(constants.ITERATION))

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.G_B.load_state_dict(checkpoint[constants.GENERATOR_KEY + "B"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.D_B.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        netGB_state_dict = self.G_B.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.GENERATOR_KEY + "B"] = netGB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "B"] = netDB_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.STYLE_TRANSFER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        # clear plots to avoid potential sudden jumps in visualization due to unstable gradients during early training
        if (epoch % 20 == 0):
            self.losses_dict[constants.G_LOSS_KEY].clear()
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].clear()
            self.losses_dict[constants.IDENTITY_LOSS_KEY].clear()
            self.losses_dict[constants.LIKENESS_LOSS_KEY].clear()
            self.losses_dict[constants.SMOOTHNESS_LOSS_KEY].clear()
            self.losses_dict[constants.G_ADV_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].clear()
            self.losses_dict[constants.D_B_FAKE_LOSS_KEY].clear()
            self.losses_dict[constants.D_B_REAL_LOSS_KEY].clear()
            self.losses_dict[constants.CYCLE_LOSS_KEY].clear()
