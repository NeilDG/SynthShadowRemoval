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

class IIDTrainer:

    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr
        self.use_bce = opts.use_bce
        self.use_mask = opts.use_mask

        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.batch_size = opts.batch_size

        num_blocks = opts.num_blocks_a
        net_config = opts.net_config_a

        if(net_config == 1):
            self.G_albedo = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif(net_config == 2):
            self.G_albedo = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        else:
            self.G_albedo = ffa.FFA(gps=3, blocks=num_blocks).to(self.gpu_device)

        num_blocks = opts.num_blocks_s
        net_config = opts.net_config_s

        if (net_config == 1):
            self.G_shading = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks).to(self.gpu_device)
        elif (net_config == 2):
            self.G_shading = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif(net_config ==3):
            self.G_shading = ffa.FFA(gps=3, blocks=num_blocks).to(self.gpu_device)
        else:
            self.G_shading = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=num_blocks, has_dropout = False).to(self.gpu_device)

        self.D_albedo = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_shading = cycle_gan.Discriminator().to(self.gpu_device)

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_albedo.parameters(), self.G_shading.parameters()), lr=self.g_lr)
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_albedo.parameters(), self.D_shading.parameters()), lr=self.d_lr)
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
        self.losses_dict[constants.RECONSTRUCTION_LOSS_KEY] = []
        self.losses_dict[constants.LPIP_LOSS_KEY] = []
        self.losses_dict[constants.SSIM_LOSS_KEY] = []
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[constants.RECONSTRUCTION_LOSS_KEY] = "RGB loss per iteration"
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

    def update_penalties(self, adv_weight, l1_weight, iid_weight, lpip_weight, ssim_weight, bce_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.iid_weight = iid_weight
        self.lpip_weight = lpip_weight
        self.ssim_weight = ssim_weight
        self.bce_weight = bce_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.IID_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.IID_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.l1_weight), file=f)
            print("LPIP weight: ", str(self.lpip_weight), file=f)
            print("SSIM weight: ", str(self.ssim_weight), file=f)
            print("BCE weight: ", str(self.bce_weight), file=f)

    def train(self, rgb_tensor, albedo_tensor, shading_tensor, train_mask):
        with amp.autocast():
            albedo_like = self.G_albedo(rgb_tensor)
            shading_like = self.G_shading(rgb_tensor)
            if (self.use_mask == 1):
                albedo_like = torch.mul(albedo_like, train_mask)

            self.D_albedo.train()
            self.D_shading.train()
            self.optimizerD.zero_grad()

            prediction = self.D_albedo(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_albedo(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_albedo(albedo_like.detach()), fake_tensor) * self.adv_weight
            D_S_real_loss = self.adversarial_loss(self.D_shading(shading_tensor), real_tensor) * self.adv_weight
            D_S_fake_loss = self.adversarial_loss(self.D_shading(shading_like.detach()), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss + D_S_real_loss + D_S_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_albedo.train()
            self.G_shading.train()
            self.optimizerG.zero_grad()

            albedo_like = self.G_albedo(rgb_tensor)
            shading_like = self.G_shading(rgb_tensor)
            if (self.use_mask == 1):
                albedo_like = torch.mul(albedo_like, train_mask)

            likeness_loss_a = self.l1_loss(albedo_like, albedo_tensor) * self.l1_weight
            lpip_loss_a = self.lpip_loss(albedo_like, albedo_tensor) * self.lpip_weight
            ssim_loss_a = self.ssim_loss(albedo_like, albedo_tensor) * self.ssim_weight
            bce_loss_val_a = self.bce_loss_term(albedo_like, albedo_tensor) * self.bce_weight

            prediction = self.D_albedo(albedo_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss_a = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            likeness_loss_s = self.l1_loss(shading_like, shading_tensor) * self.l1_weight
            lpip_loss_s = self.lpip_loss(shading_like, shading_tensor) * self.lpip_weight
            ssim_loss_s = self.ssim_loss(shading_like, shading_tensor) * self.ssim_weight
            bce_loss_val_s = self.bce_loss_term(shading_like, shading_tensor) * self.bce_weight
            rgb_loss = self.l1_loss(albedo_like * shading_like, rgb_tensor) * self.iid_weight

            prediction = self.D_shading(shading_like)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss_s = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_adv_loss_a + likeness_loss_a + lpip_loss_a + ssim_loss_a + bce_loss_val_a + \
            A_adv_loss_s + likeness_loss_s + lpip_loss_s + ssim_loss_s + bce_loss_val_s + rgb_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss_a.item() + likeness_loss_s.item())
            self.losses_dict[constants.RECONSTRUCTION_LOSS_KEY].append(rgb_loss.item())
            self.losses_dict[constants.LPIP_LOSS_KEY].append(lpip_loss_a.item() + lpip_loss_s.item())
            self.losses_dict[constants.SSIM_LOSS_KEY].append(ssim_loss_a.item() + ssim_loss_s.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss_a.item() + A_adv_loss_s.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item() + D_S_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item() + D_S_real_loss.item())

    def test(self, rgb_tensor):
        with torch.no_grad():
            albedo_like = self.G_albedo(rgb_tensor)
            shading_like = self.G_shading(rgb_tensor)
            rgb_like = albedo_like * shading_like
        return rgb_like

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict)

    def visdom_visualize(self, rgb_tensor, albedo_tensor, shading_tensor, test = False):
        with torch.no_grad():
            albedo_like = self.G_albedo(rgb_tensor) * 0.5 + 0.5
            shading_like = self.G_shading(rgb_tensor) * 0.5 + 0.5
            rgb_like = albedo_like * shading_like

            label = "Train "
            if(test == True):
                label = "Test "

            self.visdom_reporter.plot_image(rgb_tensor, label + "RGB images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, label + "RGB-Like images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_tensor, label + "Albedo images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(albedo_like, label + "Albedo-Like images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shading_tensor, label + "Shading images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(shading_like, label + "RGB2Shading images - " + constants.IID_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            albedo_like = self.G_albedo(rw_tensor)
            shading_like = self.G_shading(rw_tensor)
            rgb_like = albedo_like * shading_like
            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.IID_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rgb_like, "Real World Like images - " + constants.IID_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.G_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "S"])
        self.D_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "S"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states_checkpt(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_albedo.state_dict()
        netDA_state_dict = self.D_albedo.state_dict()

        netGS_state_dict = self.G_shading.state_dict()
        netDS_state_dict = self.D_shading.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGA_state_dict = self.G_albedo.state_dict()
        netDA_state_dict = self.D_albedo.state_dict()

        netGS_state_dict = self.G_shading.state_dict()
        netDS_state_dict = self.D_shading.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()

        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict

        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict

        torch.save(save_dict, constants.IID_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))