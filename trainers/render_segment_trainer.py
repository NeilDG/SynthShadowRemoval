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

class RenderSegmentTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce

        self.l1_loss = nn.L1Loss()

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size

        self.G_A = cycle_gan.Classifier(input_nc=3, num_classes=1, n_residual_blocks=num_blocks).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            loss = nn.L1Loss()
            return loss(pred, target)
        else:
            loss = nn.BCEWithLogitsLoss()
            return loss(pred, target)

    def l1_loss(self, pred, target):
        return self.l1_loss(pred, target)

    def update_penalties(self, adv_weight, l1_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.MAPPER_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.MAPPER_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.l1_weight), file=f)

    def train(self, a_tensor, b_tensor):
        with amp.autocast():
            self.G_A.train()
            self.optimizerG.zero_grad()

            a2b = self.G_A(a_tensor)
            likeness_loss = self.l1_loss(a2b, b_tensor) * self.l1_weight
            errG = likeness_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())

    def test(self, a_tensor):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict)

    def visdom_visualize(self, a_tensor, b_tensor, a_test, b_test):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
            test_a2b = self.G_A(a_test)

            self.visdom_reporter.plot_image(a_tensor, "Training A images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, "Training A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)
            #self.visdom_reporter.plot_image(b_tensor, "B images - " + constants.MAPPER_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(a_test, "Test A images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(test_a2b, "Test A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)
            #self.visdom_reporter.plot_image(b_test, "Test B images - " + constants.MAPPER_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rw2b = self.G_A(rw_tensor)
            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.MAPPER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.MAPPER_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()
        optimizerG_state_dict = self.optimizerG.state_dict()
        schedulerG_state_dict = self.schedulerG.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict

        torch.save(save_dict, constants.MAPPER_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_A.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        schedulerG_state_dict = self.schedulerG.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict

        torch.save(save_dict, constants.MAPPER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))