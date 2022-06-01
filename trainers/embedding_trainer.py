import os

import lpips
from model import embedding_network
from model import vanilla_cycle_gan as cycle_gan
import constants
import torch
import torch.cuda.amp as amp
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from transforms import cyclegan_transforms
from utils import plot_utils

class EmbeddingTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce

        num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size

        self.use_lpips = opts.use_lpips
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)

        #self.G_A = embedding_network.EmbeddingNetwork(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        self.G_A = embedding_network.EmbeddingNetworkFFA(blocks = num_blocks).to(self.gpu_device)
        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator

        self.transform_op = cyclegan_transforms.CycleGANTransform(opts).requires_grad_(False)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

    def initialize_dict(self):
        # what to store in visdom?
        self.EMBEDDING_KEY = "EMBEDDING_KEY"
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict[self.EMBEDDING_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []


        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[self.EMBEDDING_KEY] = "Embedding distance per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            loss = nn.MSELoss()
            return loss(pred, target)
        else:
            loss = nn.BCEWithLogitsLoss()
            return loss(pred, target)

    def likeness_loss(self, pred, target):
        if (self.use_lpips == 0):
            loss = nn.L1Loss()
            return loss(pred, target)
        else:
            result = torch.squeeze(self.lpips_loss(pred, target))
            result = torch.mean(result)
            return result

    def update_penalties(self,  adv_weight, likeness_weight, embedding_dist_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.likeness_weight = likeness_weight
        self.embedding_dist_weight = embedding_dist_weight

        print("Version: ", constants.EMBEDDING_VERSION)
        print("Learning rate for G: ", str(self.g_lr))
        print("Learning rate for D: ", str(self.d_lr))
        print("====================================")
        print("Adv weight: ", str(self.adv_weight))
        print("Likeness weight: ", str(self.likeness_weight))
        print("Embedding dist weight: ", str(self.embedding_dist_weight))

    def train(self, tensor_x, tensor_y):
        with amp.autocast():
            tensor_x = self.transform_op(tensor_x).detach()
            tensor_y = self.transform_op(tensor_y).detach()

            fake_input = self.G_A(tensor_x)
            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(tensor_x)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(tensor_x), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A(fake_input.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            clean_like = self.G_A(tensor_x)
            likeness_loss = self.likeness_loss(clean_like, tensor_x) * self.likeness_weight

            #reduce embedding distance loss
            tensor_x_embedding, _, _, _ = self.G_A.get_embedding(tensor_x)
            tensor_y_embedding, _, _, _ = self.G_A.get_embedding(tensor_y)
            # tensor_x_embedding = torch.mean(tensor_x_embedding)
            # tensor_y_embedding = torch.mean(tensor_y_embedding)
            embedding_dist_loss = self.likeness_loss(tensor_x_embedding, tensor_y_embedding) * self.embedding_dist_weight

            prediction = self.D_A(fake_input)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_adv_loss + likeness_loss + embedding_dist_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[self.EMBEDDING_KEY].append(embedding_dist_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def test(self, tensor_x):
        with torch.no_grad():
            return self.G_A(tensor_x)

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.EMBEDDING_CHECKPATH)

    def visdom_visualize(self, tensor_x, tensor_y, label ="Train"):
        with torch.no_grad():
            # report to visdom
            if(label == "Train"):
                tensor_x = self.transform_op(tensor_x).detach()
                tensor_y = self.transform_op(tensor_y).detach()

            fake_a = self.G_A(tensor_x)
            fake_b = self.G_A(tensor_y)

            self.visdom_reporter.plot_image(tensor_x, str(label) + " A images - " + constants.EMBEDDING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(fake_a, str(label) + "  Reconstructed A images - " + constants.EMBEDDING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(tensor_y, str(label) + "  B images - " + constants.EMBEDDING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(fake_b, str(label) + "  Reconstructed B images - " + constants.EMBEDDING_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rw2b = self.G_A(rw_tensor)
            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.EMBEDDING_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.EMBEDDING_VERSION + constants.ITERATION)


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

        torch.save(save_dict, constants.EMBEDDING_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
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

        torch.save(save_dict, constants.EMBEDDING_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))