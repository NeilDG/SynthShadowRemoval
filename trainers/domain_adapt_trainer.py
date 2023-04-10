# -*- coding: utf-8 -*-
# DA trainer based on
# Gonzalez-Garcia, A., Van De Weijer, J., & Bengio, Y. (2018).
# Image-to-image translation for cross-domain disentanglement. Advances in neural information processing systems, 31.
import kornia
from model import translator_gan
from model import vanilla_cycle_gan as cycle_gan
import global_config
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from utils import plot_utils
from utils import tensor_utils
from losses import ssim_loss
import lpips

class DomainAdaptIterationTable():
    def __init__(self):
        self.iteration_table = {}

        iteration = 1
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 2
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 3
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, feature_l1_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 4
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, feature_l1_weight=1.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 5
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, feature_l1_weight=10.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 6
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=1.0, feature_l1_weight=10.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 7
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, feature_l1_weight=10.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 8
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=10.0, feature_l1_weight=10.0, lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 9
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 10
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 11
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 12
        self.iteration_table[str(iteration)] = IterationParameters(iteration, l1_weight=0.0, feature_l1_weight=1.0, lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

    def get_version(self, iteration):
        return self.iteration_table[str(iteration)]


class IterationParameters():
    def __init__(self, iteration, l1_weight, feature_l1_weight, lpip_weight, cycle_weight, adv_weight, is_bce,
                 num_blocks):
        self.iteration = iteration
        self.l1_weight = l1_weight
        self.feature_l1_weight = feature_l1_weight
        self.lpip_weight = lpip_weight
        self.cycle_weight = cycle_weight
        self.adv_weight = adv_weight
        self.is_bce = is_bce
        self.num_blocks = num_blocks

class DomainAdaptTrainer():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

        self.iteration = opts.iteration
        it_params = DomainAdaptIterationTable().get_version(self.iteration)
        self.use_bce = it_params.is_bce
        num_blocks = it_params.num_blocks
        self.batch_size = opts.load_size

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.update_penalties(it_params.adv_weight, it_params.l1_weight, it_params.feature_l1_weight, it_params.lpip_weight, it_params.cycle_weight)

        self.G_X2Y = translator_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        self.G_Y2X = translator_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)

        self.G_X2Y_D = translator_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        self.G_Y2X_D = translator_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)

        self.D_X = cycle_gan.Discriminator(input_nc=3, use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_Y = cycle_gan.Discriminator(input_nc=3, use_bce=self.use_bce).to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_X2Y.parameters(), self.G_Y2X.parameters(), self.G_X2Y_D.parameters(), self.G_Y2X_D.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_X.parameters(), self.D_Y.parameters()), lr=self.d_lr)
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
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.CYCLE_LOSS_KEY] = []
        self.FEATURE_L1_LOSS_KEY = "FEATURE_L1_LOSS_KEY"
        self.losses_dict[self.FEATURE_L1_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"
        self.caption_dict[self.FEATURE_L1_LOSS_KEY] = "Feature L1 loss per iteration"

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

    def update_penalties(self, adv_weight, l1_weight, feature_l1_weight, lpip_weight, cycle_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.feature_l1_weight = feature_l1_weight
        self.lpip_weight = lpip_weight
        self.cycle_weight = cycle_weight

        print("Adv weight: ", self.adv_weight)
        print("L1 weight: ", self.l1_weight)
        print("Feature L1 weight: ", self.feature_l1_weight)
        print("LPIP weight: ", self.lpip_weight)
        print("Cycle weight: ", self.cycle_weight)

    def train(self, tensor_x, tensor_y):
        with amp.autocast():
            self.optimizerD.zero_grad()

            #train discriminator
            # X --> Y translation
            x2y = self.G_X2Y_D.get_decoding(self.G_Y2X_D.get_encoding(tensor_y))
            # x2y = self.G_X2Y(tensor_x)
            self.D_Y.train()
            prediction = self.D_Y(x2y)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_Y_real_loss = self.adversarial_loss(self.D_Y(tensor_y), real_tensor) * self.adv_weight
            D_Y_fake_loss = self.adversarial_loss(self.D_Y(x2y.detach()), fake_tensor) * self.adv_weight

            # Y --> X translation
            y2x = self.G_Y2X_D.get_decoding(self.G_X2Y_D.get_encoding(tensor_x))
            # y2x = self.G_Y2X(tensor_y)
            self.D_X.train()
            prediction = self.D_X(y2x)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_X_real_loss = self.adversarial_loss(self.D_X(tensor_x), real_tensor) * self.adv_weight
            D_X_fake_loss = self.adversarial_loss(self.D_X(y2x.detach()), fake_tensor) * self.adv_weight
            errD = D_Y_real_loss + D_Y_fake_loss + D_X_real_loss + D_X_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            #train generator
            # X --> Y translation
            self.G_X2Y.train()
            self.G_Y2X.train()
            self.G_X2Y_D.train()
            self.G_Y2X_D.train()
            self.optimizerG.zero_grad()

            x2y_l1_loss = self.l1_loss(self.G_X2Y(tensor_x), tensor_y) * self.l1_weight
            x2y_lpip_loss = self.lpip_loss(self.G_X2Y(tensor_x), tensor_y) * self.lpip_weight

            # Y --> X translation
            y2x_l1_loss = self.l1_loss(self.G_Y2X(tensor_y), tensor_x) * self.l1_weight
            y2x_lpip_loss = self.lpip_loss(self.G_Y2X(tensor_y), tensor_x) * self.lpip_weight

            #reduce feature distance
            feature_l1_loss = self.l1_loss(self.G_X2Y_D.get_encoding(tensor_y), self.G_Y2X_D.get_encoding(tensor_x)) * self.feature_l1_weight

            #reduce exclusive feature reconstruction
            #X --> Y translation
            x2y_shared = self.G_X2Y_D.get_decoding(self.G_Y2X_D.get_encoding(tensor_y))
            prediction = self.D_Y(x2y_shared)
            real_tensor = torch.ones_like(prediction)
            x2y_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            #Y --> X translation
            y2x_shared = self.G_Y2X_D.get_decoding(self.G_X2Y_D.get_encoding(tensor_x))
            prediction = self.D_X(y2x_shared)
            real_tensor = torch.ones_like(prediction)
            y2x_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            #feature cycle consistency
            x2y_cycle_loss = self.l1_loss(self.G_Y2X(self.G_X2Y(tensor_x)), tensor_x) * self.cycle_weight
            y2X_cycle_loss = self.l1_loss(self.G_X2Y(self.G_Y2X(tensor_y)), tensor_y) * self.cycle_weight

            errG = x2y_l1_loss + y2x_l1_loss + feature_l1_loss + x2y_adv_loss + y2x_adv_loss + x2y_cycle_loss + y2X_cycle_loss + \
                x2y_lpip_loss + y2x_lpip_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(x2y_l1_loss.item() + y2x_l1_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(x2y_adv_loss.item() + y2x_adv_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(x2y_cycle_loss.item() + y2X_cycle_loss.item())
        self.losses_dict[self.FEATURE_L1_LOSS_KEY].append(feature_l1_loss.item())

    def test(self, tensor_x, tensor_y):
        with torch.no_grad():
            x2y = self.G_X2Y(tensor_x)
            y2x = self.G_Y2X(tensor_y)
            return x2y, y2x

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.STYLE_TRANSFER_CHECKPATH)

    def visdom_visualize(self, tensor_x, tensor_y, label = "Training"):
        with torch.no_grad():
            x2y = self.G_X2Y(tensor_x)
            y2x = self.G_Y2X(tensor_y)

            self.visdom_reporter.plot_image(tensor_x, str(label) + " Input X Images - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(y2x, str(label) + " X Image Reconstruction - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(tensor_y, str(label) + " Input Y Images - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(x2y, str(label) + " Y Image Reconstruction - " + constants.STYLE_TRANSFER_VERSION + global_config.ITERATION)

    def load_saved_state(self, checkpoint):
        self.G_X2Y.load_state_dict(checkpoint[constants.GENERATOR_KEY + "X2Y"])
        self.G_Y2X.load_state_dict(checkpoint[constants.GENERATOR_KEY + "Y2X"])
        self.G_X2Y_D.load_state_dict(checkpoint[constants.GENERATOR_KEY + "X2Y_D"])
        self.G_Y2X_D.load_state_dict(checkpoint[constants.GENERATOR_KEY + "Y2X_D"])

        self.D_X.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "DX"])
        self.D_Y.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "DY"])

        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + global_config.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGX2Y_state_dict = self.G_X2Y.state_dict()
        netGY2X_state_dict = self.G_Y2X.state_dict()
        netGX2Y_D_state_dict = self.G_X2Y_D.state_dict()
        netGY2X_D_state_dict = self.G_Y2X_D.state_dict()

        netDX_state_dict = self.D_X.state_dict()
        netDY_state_dict = self.D_Y.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "X2Y"] = netGX2Y_state_dict
        save_dict[constants.GENERATOR_KEY + "Y2X"] = netGY2X_state_dict
        save_dict[constants.GENERATOR_KEY + "X2Y_D"] = netGX2Y_D_state_dict
        save_dict[constants.GENERATOR_KEY + "Y2X_D"] = netGY2X_D_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DX"] = netDX_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "DY"] = netDY_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.STYLE_TRANSFER_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGX2Y_state_dict = self.G_X2Y.state_dict()
        netGY2X_state_dict = self.G_Y2X.state_dict()
        netGX2Y_D_state_dict = self.G_X2Y_D.state_dict()
        netGY2X_D_state_dict = self.G_Y2X_D.state_dict()

        netDX_state_dict = self.D_X.state_dict()
        netDY_state_dict = self.D_Y.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "X2Y"] = netGX2Y_state_dict
        save_dict[constants.GENERATOR_KEY + "Y2X"] = netGY2X_state_dict
        save_dict[constants.GENERATOR_KEY + "X2Y_D"] = netGX2Y_D_state_dict
        save_dict[constants.GENERATOR_KEY + "Y2X_D"] = netGY2X_D_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DX"] = netDX_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DY"] = netDY_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.STYLE_TRANSFER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))



