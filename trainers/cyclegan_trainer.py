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
from lpips import lpips

import constants
from config import iid_server_config
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from model import usi3d_gan
from model import new_style_transfer_gan
from trainers import early_stopper
from utils import plot_utils
from utils import pytorch_colors
from transforms import cyclegan_transforms
from model.modules import image_pool

class DomainAdaptIterationTable():
    def __init__(self):
        self.iteration_table = {}

        #disc_mode = 0, 1, 2. 0 = MSELoss, 1 = BCE Loss, 2 = L1 Loss

        iteration = 1
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 2
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 3
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 4
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 5
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, id_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 6
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, id_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 7
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, id_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 8
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, id_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 9
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 10
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 11
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 12
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, id_weight=0.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 13
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, id_weight=0.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 14
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, id_weight=0.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

        iteration = 15
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, id_weight=0.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=0)

        iteration = 16
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, id_weight=0.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, disc_mode=1)

    def get_version(self, iteration):
        return self.iteration_table[str(iteration)]


class IterationParameters():
    def __init__(self, iteration, l1_weight, id_weight, lpip_weight, cycle_weight, adv_weight, disc_mode):
        self.iteration = iteration
        self.l1_weight = l1_weight
        self.id_weight = id_weight
        self.lpip_weight = lpip_weight
        self.cycle_weight = cycle_weight
        self.adv_weight = adv_weight
        self.disc_mode = disc_mode

class CycleGANTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

        self.iteration = opts.iteration
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_style_transfer_config_from_version()

        net_config = network_config["net_config"]
        num_blocks = network_config["num_blocks"]
        norm_mode = network_config["norm_mode"]
        self.batch_size = network_config["batch_size"]
        self.load_size = network_config["load_size"]

        it_params = DomainAdaptIterationTable().get_version(self.iteration)
        self.disc_mode = it_params.disc_mode

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.update_penalties(it_params.adv_weight, it_params.id_weight, it_params.l1_weight, it_params.lpip_weight, it_params.cycle_weight)

        if(net_config == 1):
            print("Using Cycle GAN")
            self.G_X2Y = cycle_gan.Generator(n_residual_blocks=num_blocks, has_dropout=False, norm=norm_mode).to(self.gpu_device)
            self.G_Y2X = cycle_gan.Generator(n_residual_blocks=num_blocks, has_dropout=False, norm=norm_mode).to(self.gpu_device)
        elif(net_config == 2):
            if(norm_mode == "instance"):
                print("Using U-Net GAN - Instance Norm")
                norm_layer = nn.InstanceNorm2d
            else:
                print("Using U-Net GAN - Batch Norm")
                norm_layer = nn.BatchNorm2d
            self.G_X2Y = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks, norm_layer=norm_layer).to(self.gpu_device)
            self.G_Y2X = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks, norm_layer=norm_layer).to(self.gpu_device)
        else:
            print("Using AdainGEN")
            params = {'dim': 64,  # number of filters in the bottommost layer
                      'mlp_dim': 256,  # number of filters in MLP
                      'style_dim': 8,  # length of style code
                      'n_layer': 3,  # number of layers in feature merger/splitor
                      'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,  # number of downsampling layers in content encoder
                      'n_res': num_blocks,  # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            self.G_X2Y = usi3d_gan.AdaINGen(input_dim=3, output_dim=3, params=params).to(self.gpu_device)
            self.G_Y2X = usi3d_gan.AdaINGen(input_dim=3, output_dim=3, params=params).to(self.gpu_device)


        self.D_X = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_Y = cycle_gan.Discriminator().to(self.gpu_device)

        self.D_X_pool = image_pool.ImagePool(50)
        self.D_Y_pool = image_pool.ImagePool(50)

        self.transform_op = cyclegan_transforms.CycleGANTransform(network_config["patch_size"]).requires_grad_(False)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_X2Y.parameters(), self.G_Y2X.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=self.d_lr)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=10000, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=10000, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_style_transfer"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.NETWORK_VERSION = sc_instance.get_version_config("style_transfer_version", "style_transfer_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

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

    def update_penalties(self, adv_weight, id_weight, likeness_weight, lpip_weight, cycle_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.id_weight = id_weight
        self.likeness_weight = likeness_weight
        self.lpip_weight = lpip_weight
        self.cycle_weight = cycle_weight

        print("Adv weight: ", self.adv_weight)
        print("Identity weight: ", self.id_weight)
        print("Likeness weight: ", self.likeness_weight)
        print("LPIP weight: ", self.lpip_weight)
        print("Cycle weight: ", self.cycle_weight)
        print("Disc Mode: ", self.disc_mode)

    def adversarial_loss(self, pred, target):
        if(self.disc_mode == 0):
            return self.mse_loss(pred, target)
        elif(self.disc_mode == 1):
            return self.bce_loss(pred, target)
        else:
            return self.l1_loss(pred, target)

    def identity_loss(self, pred, target):
        return self.l1_loss(pred, target)

    def cycle_loss(self, pred, target):
        return self.l1_loss(pred, target)

    def likeness_loss(self, pred, target):
        return self.l1_loss(pred, target)

        # loss = vgg.VGGPerceptualLoss().to(self.gpu_device)
        # return loss(pred, target)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    # def accumulate_input(self, input_x, input_y):
    #     if(self.input_x == None):
    #         self.input_x = input_x
    #     else:
    #         self.input_x = torch.cat([self.input_x, input_x], 0)
    #
    #     if (self.input_y == None):
    #         self.input_y = input_y
    #     else:
    #         self.input_y = torch.cat([self.input_y, input_y], 0)
    #
    #     self.accumulated_size = max(np.shape(self.input_x)[0], np.shape(self.input_y)[0])
    #     print("Accumulating input: ", np.shape(self.input_x), np.shape(self.input_y))
    #
    # def clear_input(self):
    #     self.input_x = None
    #     self.input_y = None
    #     self.accumulated_size = 0

    def train(self, epoch, iteration, tensor_x, tensor_y, batch):
        with amp.autocast():
            tensor_x = self.transform_op(tensor_x).detach()
            tensor_y = self.transform_op(tensor_y).detach()

            accum_batch_size = np.shape(tensor_x)[0] * (batch + 1)
            # print("Current tensor size: ", np.shape(tensor_x), np.shape(tensor_y), " Accum batch size: ", accum_batch_size)


            # optimize D-----------------------
            y_like = self.G_X2Y(tensor_x)
            x_like = self.G_Y2X(tensor_y)

            self.D_X.train()
            self.D_Y.train()
            self.optimizerD.zero_grad()

            prediction = self.D_X(tensor_x)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_X(tensor_x), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_X_pool.query(self.D_X(x_like.detach())), fake_tensor) * self.adv_weight

            prediction = self.D_Y(tensor_y)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.adversarial_loss(self.D_Y(tensor_y), real_tensor) * self.adv_weight
            D_B_fake_loss = self.adversarial_loss(self.D_Y_pool.query(self.D_Y(y_like.detach())), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0 and accum_batch_size % self.batch_size == 0):
                self.schedulerD.step(errD)
                self.fp16_scaler.step(self.optimizerD)

            # optimize G-----------------------
            self.G_X2Y.train()
            self.G_Y2X.train()
            self.optimizerG.zero_grad()

            identity_like = self.G_X2Y(tensor_y)
            y_like = self.G_X2Y(tensor_x)

            A_identity_loss = self.identity_loss(identity_like, tensor_y) * self.id_weight
            A_likeness_loss = self.likeness_loss(y_like, tensor_y) * self.likeness_weight
            A_lpip_loss = self.lpip_loss(y_like, tensor_y) * self.lpip_weight
            A_cycle_loss = self.cycle_loss(self.G_Y2X(self.G_X2Y(tensor_x)), tensor_x) * self.cycle_weight
            prediction = self.D_Y(y_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            identity_like = self.G_Y2X(tensor_x)
            x_like = self.G_Y2X(tensor_y)

            B_identity_loss = self.identity_loss(identity_like, tensor_x) * self.id_weight
            B_likeness_loss = self.likeness_loss(x_like, tensor_x) * self.likeness_weight
            B_lpip_loss = self.lpip_loss(x_like, tensor_x) * self.lpip_weight
            B_cycle_loss = self.cycle_loss(self.G_X2Y(self.G_Y2X(tensor_y)), tensor_y) * self.cycle_weight
            prediction = self.D_X(x_like)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_identity_loss + B_identity_loss + A_likeness_loss + B_likeness_loss + A_lpip_loss + B_lpip_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss
            self.fp16_scaler.scale(errG).backward()

            if(accum_batch_size % self.batch_size == 0):
                self.schedulerG.step(errG)
                self.fp16_scaler.step(self.optimizerG)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
                self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
                self.losses_dict[constants.IDENTITY_LOSS_KEY].append(A_identity_loss.item() + B_identity_loss.item())
                self.losses_dict[constants.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
                self.losses_dict[constants.SMOOTHNESS_LOSS_KEY].append(A_lpip_loss.item() + B_lpip_loss.item())
                self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
                self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
                self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
                self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
                self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
                self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())

            x2y, _ = self.test(tensor_x, tensor_y)
            self.stopper_method.register_metric(x2y, tensor_y, epoch)
            self.stop_result = self.stopper_method.test(epoch)

            if (self.stopper_method.has_reset()):
                self.save_states(epoch, iteration, False)

    def is_stop_condition_met(self):
        return self.stopper_method.did_stop_condition_met()

    def test(self, tensor_x, tensor_y):
        with torch.no_grad():
            x2y = self.G_X2Y(tensor_x)
            y2x = self.G_Y2X(tensor_y)
            return x2y, y2x

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.style_transfer_version)

    def visdom_visualize(self, tensor_x, tensor_y, label="Train"):
        with torch.no_grad():
            if(label == "Train"):
                tensor_x = self.transform_op(tensor_x).detach()
                tensor_y = self.transform_op(tensor_y).detach()

            x2y = self.G_X2Y(tensor_x)
            y2x = self.G_Y2X(tensor_y)

            x2y2x = self.G_Y2X(x2y)
            y2x2y = self.G_X2Y(y2x)

            self.visdom_reporter.plot_image(tensor_x, str(label) + " Input X Images - " + constants.style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(x2y2x, str(label) + " Input X Cycle - " + constants.style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(x2y, str(label) + " X2Y Transfer " + constants.style_transfer_version + str(self.iteration))

            self.visdom_reporter.plot_image(tensor_y, str(label) + " Input Y Images - " + constants.style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(y2x2y, str(label) + " Input Y Cycle - " + constants.style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(y2x, str(label) + " Y2X Transfer - " + constants.style_transfer_version + str(self.iteration))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
            print("Loaded network: ", self.NETWORK_CHECKPATH)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("Loaded network: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_style_transfer", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_X2Y.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
            self.G_Y2X.load_state_dict(checkpoint[constants.GENERATOR_KEY + "B"])
            self.D_X.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
            self.D_Y.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "B"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGA_state_dict = self.G_X2Y.state_dict()
        netGB_state_dict = self.G_Y2X.state_dict()
        netDA_state_dict = self.D_X.state_dict()
        netDB_state_dict = self.D_Y.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.GENERATOR_KEY + "B"] = netGB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "B"] = netDB_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
