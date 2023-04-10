import itertools
import os
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.utils as vutils
import global_config
from utils import plot_utils
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from custom_losses import ssim_loss
from custom_losses import vgg_loss_model as vgg_loss
import lpips
from model import embedding_network

class IterationTable():
    def __init__(self):
        self.iteration_table = {}

        iteration = 1
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 2
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 3
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=1.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 4
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=1.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 5
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=1.0, likeness_weight=1.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 6
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=1.0, likeness_weight=1.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        iteration = 7
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=1.0, likeness_weight=0.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
                                                                   num_blocks=6)

        iteration = 8
        self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=1.0, likeness_weight=0.0, feature_l1_weight=1.0,
                                                                   lpip_weight=0.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
                                                                   num_blocks=6)

        # iteration = 9
        # self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
        #                                                            lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
        #                                                            num_blocks=6)
        #
        # iteration = 10
        # self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
        #                                                            lpip_weight=1.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
        #                                                            num_blocks=6)
        #
        # iteration = 11
        # self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
        #                                                            lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, is_bce=0,
        #                                                            num_blocks=6)
        #
        # iteration = 12
        # self.iteration_table[str(iteration)] = IterationParameters(iteration, identity_weight=0.0, likeness_weight=0.0, feature_l1_weight=1.0,
        #                                                            lpip_weight=10.0, cycle_weight=10.0, adv_weight=1.0, is_bce=1,
        #                                                            num_blocks=6)

    def get_version(self, iteration):
        return self.iteration_table[str(iteration)]


class IterationParameters():
    def __init__(self, iteration, identity_weight, likeness_weight, feature_l1_weight, lpip_weight, cycle_weight, adv_weight, is_bce,
                 num_blocks):
        self.iteration = iteration
        self.identity_weight = identity_weight
        self.likeness_weight = likeness_weight
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
        it_params = IterationTable().get_version(self.iteration)
        self.use_bce = it_params.is_bce
        num_blocks = it_params.num_blocks
        self.batch_size = opts.load_size

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = ssim_loss.SSIM()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.update_penalties(it_params)

        net_config = opts.net_config

        if (net_config == 1):
            print("Using vanilla cycle GAN")
            self.G_A = cycle_gan.Generator(n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
            self.G_B = cycle_gan.Generator(n_residual_blocks=num_blocks, has_dropout=False).to(self.gpu_device)
        else:
            print("Using U-Net GAN")
            self.G_A = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
            self.G_B = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=num_blocks).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator(use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_B = cycle_gan.Discriminator(use_bce=self.use_bce).to(self.gpu_device)

        #for latent space reduction
        self.G_A2Z = embedding_network.EmbeddingNetworkFFA(blocks=4).to(self.gpu_device)
        self.D_Z = cycle_gan.Discriminator(use_bce=self.use_bce).to(self.gpu_device)  # use CycleGAN's discriminator

        self.visdom_reporter = plot_utils.VisdomReporter()
        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=self.g_lr)
        self.optimizerG_Z = torch.optim.Adam(itertools.chain(self.G_A2Z.parameters()), lr=self.g_lr)

        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=self.d_lr)
        self.optimizerD_Z = torch.optim.Adam(itertools.chain(self.D_Z.parameters()), lr=self.d_lr)

        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000, threshold=0.00005)
        self.schedulerG_Z = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_Z, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_Z = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_Z, patience=100000 / self.batch_size, threshold=0.00005)
        self.initialize_dict()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision
        self.fp16_scaler_z = amp.GradScaler()

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[constants.G_LOSS_KEY] = []
        self.losses_dict[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[constants.IDENTITY_LOSS_KEY] = []
        self.losses_dict[constants.LIKENESS_LOSS_KEY] = []
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
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"
        self.caption_dict[constants.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[constants.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"
        self.caption_dict[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"

        #for embedding network
        self.losses_dict_z = {}
        self.losses_dict_z[constants.G_LOSS_KEY] = []
        self.losses_dict_z[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_z[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_z[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_z[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_z[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict_z = {}
        self.caption_dict_z[constants.G_LOSS_KEY] = "G_Z loss per iteration"
        self.caption_dict_z[constants.D_OVERALL_LOSS_KEY] = "D_Z loss per iteration"
        self.caption_dict_z[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict_z[constants.G_ADV_LOSS_KEY] = "G_Z adv loss per iteration"
        self.caption_dict_z[constants.D_A_FAKE_LOSS_KEY] = "D_Z fake loss per iteration"
        self.caption_dict_z[constants.D_A_REAL_LOSS_KEY] = "D_Z real loss per iteration"

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.l1_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def identity_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def cycle_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)
        # loss = ssim_loss.SSIM()
        # return 1 - loss(pred, target)

    def likeness_loss(self, pred, target):
        loss = nn.L1Loss()
        return loss(pred, target)

    def perceptual_loss(self, pred, target):
        loss = vgg_loss.VGGPerceptualLoss().to(self.gpu_device)
        return loss(pred, target)

    def bce_loss_term(self, pred, target):
        return self.bce_loss(pred, target)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def update_penalties(self, it_params:IterationParameters):
        # what penalties to use for losses?
        self.adv_weight = it_params.adv_weight
        self.id_weight = it_params.identity_weight
        self.likeness_weight = it_params.likeness_weight
        self.cycle_weight = it_params.cycle_weight
        self.feature_l1_weight = it_params.feature_l1_weight
        self.lpip_weight = it_params.lpip_weight

        print("Adv weight: ", self.adv_weight)
        print("ID weight: ", self.id_weight)
        print("Likeness weight: ", self.likeness_weight)
        print("Cycle weight: ", self.cycle_weight)
        print("Feature L1 weight: ", self.feature_l1_weight)
        print("LPIP weight: ", self.lpip_weight)

    def train(self, a_tensor, b_tensor):
        with amp.autocast():
            self.train_transfer(a_tensor, b_tensor)
            self.train_embedding(a_tensor)
    
    def train_transfer(self, a_tensor, b_tensor):
        clean_like = self.G_A(a_tensor)
        dirty_like = self.G_B(b_tensor)

        self.D_A.train()
        self.D_B.train()
        self.optimizerD.zero_grad()

        prediction = self.D_A(b_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)

        D_A_real_loss = self.adversarial_loss(self.D_A(b_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_A(clean_like.detach()), fake_tensor) * self.adv_weight

        prediction = self.D_B(a_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)

        D_B_real_loss = self.adversarial_loss(self.D_B(a_tensor), real_tensor) * self.adv_weight
        D_B_fake_loss = self.adversarial_loss(self.D_B(dirty_like.detach()), fake_tensor) * self.adv_weight

        errD = D_A_real_loss + D_A_fake_loss + D_B_real_loss + D_B_fake_loss
        self.fp16_scaler.scale(errD).backward()
        if (self.fp16_scaler.scale(errD).item() > 0.2):
            self.fp16_scaler.step(self.optimizerD)
            self.schedulerD.step(errD)

        self.G_A.train()
        self.G_B.train()
        self.optimizerG.zero_grad()

        identity_like = self.G_A(b_tensor)
        clean_like = self.G_A(a_tensor)

        A_identity_loss = self.identity_loss(identity_like, b_tensor) * self.id_weight
        A_likeness_loss = self.likeness_loss(clean_like, b_tensor) * self.likeness_weight
        A_cycle_loss = self.cycle_loss(self.G_B(self.G_A(a_tensor)), a_tensor) * self.cycle_weight

        identity_like = self.G_B(a_tensor)
        dirty_like = self.G_B(b_tensor)
        B_identity_loss = self.identity_loss(identity_like, a_tensor) * self.id_weight
        B_likeness_loss = self.likeness_loss(dirty_like, a_tensor) * self.likeness_weight
        B_cycle_loss = self.cycle_loss(self.G_A(self.G_B(b_tensor)), b_tensor) * self.cycle_weight

        prediction = self.D_A(clean_like)
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

        prediction = self.D_B(dirty_like)
        real_tensor = torch.ones_like(prediction)
        B_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

        errG = A_identity_loss + B_identity_loss + A_likeness_loss + B_likeness_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss

        self.fp16_scaler.scale(errG).backward()
        self.fp16_scaler.step(self.optimizerG)
        self.schedulerG.step(errG)
        self.fp16_scaler.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict[constants.G_LOSS_KEY].append(errG.item())
        self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict[constants.IDENTITY_LOSS_KEY].append(A_identity_loss.item() + B_identity_loss.item())
        self.losses_dict[constants.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
        self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
        self.losses_dict[constants.D_B_FAKE_LOSS_KEY].append(D_B_fake_loss.item())
        self.losses_dict[constants.D_B_REAL_LOSS_KEY].append(D_B_real_loss.item())
        self.losses_dict[constants.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())

    def train_embedding(self, input_tensor):
        fake_input = self.G_A2Z(input_tensor)
        self.D_Z.train()
        self.optimizerD_Z.zero_grad()

        prediction = self.D_Z(input_tensor)
        real_tensor = torch.ones_like(prediction)
        fake_tensor = torch.zeros_like(prediction)

        D_A_real_loss = self.adversarial_loss(self.D_Z(input_tensor), real_tensor) * self.adv_weight
        D_A_fake_loss = self.adversarial_loss(self.D_Z(fake_input.detach()), fake_tensor) * self.adv_weight

        errD = D_A_real_loss + D_A_fake_loss

        self.fp16_scaler_z.scale(errD).backward()
        if (self.fp16_scaler_z.scale(errD).item() > 0.2):
            self.fp16_scaler_z.step(self.optimizerD_Z)
            self.schedulerD_Z.step(errD)

        self.G_A2Z.train()
        self.optimizerG_Z.zero_grad()

        clean_like = self.G_A2Z(input_tensor)
        likeness_loss = self.likeness_loss(clean_like, input_tensor) * self.feature_l1_weight

        prediction = self.D_Z(fake_input)
        real_tensor = torch.ones_like(prediction)
        A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

        errG = A_adv_loss + likeness_loss

        self.fp16_scaler_z.scale(errG).backward()
        self.fp16_scaler_z.step(self.optimizerG_Z)
        self.schedulerG.step(errG)
        self.fp16_scaler_z.update()

        # what to put to losses dict for visdom reporting?
        self.losses_dict_z[constants.G_LOSS_KEY].append(likeness_loss.item())
        self.losses_dict_z[constants.D_OVERALL_LOSS_KEY].append(errD.item())
        self.losses_dict_z[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
        self.losses_dict_z[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
        self.losses_dict_z[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
        self.losses_dict_z[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def test(self, tensor_a, tensor_b):
        with torch.no_grad():
            a2b = self.G_A(tensor_a)
            b2a = self.G_B(tensor_b)
            return a2b, b2a

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.STYLE_TRANSFER_CHECKPATH)
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_z", iteration, self.losses_dict_z, self.caption_dict_z, constants.STYLE_TRANSFER_CHECKPATH)

    def visdom_visualize(self, a_tensor, b_tensor, label="Training"):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)
            b2a = self.G_B(b_tensor)

            a_embedding = self.G_A2Z(a_tensor)
            b_embedding = self.G_A2Z(b_tensor)

            self.visdom_reporter.plot_image(a_tensor, str(label) + " Input A Images - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, str(label) + " A Image Reconstruction - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a_embedding, str(label) + " A Image Embedding - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)

            self.visdom_reporter.plot_image(b_tensor, str(label) + " Input B Images - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b2a, str(label) + " B Image Reconstruction - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_embedding, str(label) + " B Image Embedding - " + constants.STYLE_TRANSFER_VERSION + constants.ITERATION)

    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A2B"])
        self.G_B.load_state_dict(checkpoint[constants.GENERATOR_KEY + "B2A"])
        self.G_A2Z.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A2Z"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "DX"])
        self.D_B.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "DY"])
        self.D_Z.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "DZ"])

        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerG_Z.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD_Z.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerG_Z.load_state_dict(checkpoint[constants.GENERATOR_KEY + "schedulerZ"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])
        self.schedulerD_Z.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "schedulerZ"])

    def save_states_checkpt(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA2B_state_dict = self.G_A.state_dict()
        netGB2A_state_dict = self.G_B.state_dict()
        netGA2Z_state_dict = self.G_A2Z.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerGZ_state_dict = self.optimizerG_Z.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        optimizerDZ_state_dict = self.optimizerD_Z.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerGZ_state_dict = self.schedulerG_Z.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()
        schedulerDZ_state_dict = self.schedulerD_Z.state_dict()

        save_dict[constants.GENERATOR_KEY + "A2B"] = netGA2B_state_dict
        save_dict[constants.GENERATOR_KEY + "B2A"] = netGB2A_state_dict
        save_dict[constants.GENERATOR_KEY + "A2Z"] = netGA2Z_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DX"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DY"] = netDB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DZ"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerDZ_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict
        save_dict[constants.GENERATOR_KEY + "schedulerZ"] = schedulerGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "schedulerZ"] = schedulerDZ_state_dict

        torch.save(save_dict, constants.STYLE_TRANSFER_CHECKPATH + ".checkpt")
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: last_metric}
        netGA2B_state_dict = self.G_A.state_dict()
        netGB2A_state_dict = self.G_B.state_dict()
        netGA2Z_state_dict = self.G_A2Z.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()
        netDZ_state_dict = self.D_Z.state_dict()

        optimizerG_state_dict = self.optimizerG.state_dict()
        optimizerGZ_state_dict = self.optimizerG_Z.state_dict()
        optimizerD_state_dict = self.optimizerD.state_dict()
        optimizerDZ_state_dict = self.optimizerD_Z.state_dict()

        schedulerG_state_dict = self.schedulerG.state_dict()
        schedulerGZ_state_dict = self.schedulerG_Z.state_dict()
        schedulerD_state_dict = self.schedulerD.state_dict()
        schedulerDZ_state_dict = self.schedulerD_Z.state_dict()

        save_dict[constants.GENERATOR_KEY + "A2B"] = netGA2B_state_dict
        save_dict[constants.GENERATOR_KEY + "B2A"] = netGB2A_state_dict
        save_dict[constants.GENERATOR_KEY + "A2Z"] = netGA2Z_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DX"] = netDA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DY"] = netDB_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "DZ"] = netDZ_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY] = optimizerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY] = optimizerD_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerDZ_state_dict

        save_dict[constants.GENERATOR_KEY + "scheduler"] = schedulerG_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler"] = schedulerD_state_dict
        save_dict[constants.GENERATOR_KEY + "schedulerZ"] = schedulerGZ_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "schedulerZ"] = schedulerDZ_state_dict

        torch.save(save_dict, constants.STYLE_TRANSFER_CHECKPATH)
        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

        