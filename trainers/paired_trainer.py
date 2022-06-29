# -*- coding: utf-8 -*-
# Paired trainer used for training.

from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn

from model.modules import image_pool
from utils import plot_utils
import lpips

class IterationTable():
    def __init__(self):
        #initialize table
        self.iteration_table = {}

        iteration = 1
        self.iteration_table[iteration] = IterationParameters(iteration, [10.0, 0.0], is_bce=0)

        iteration = 2
        self.iteration_table[iteration] = IterationParameters(iteration, [10.0, 0.0], is_bce=1)

        iteration = 3
        self.iteration_table[iteration] = IterationParameters(iteration, [0.0, 10.0], is_bce=0)

        iteration = 4
        self.iteration_table[iteration] = IterationParameters(iteration, [0.0, 10.0], is_bce=1)

        iteration = 5
        self.iteration_table[iteration] = IterationParameters(iteration, [10.0, 10.0], is_bce=0)

        iteration = 6
        self.iteration_table[iteration] = IterationParameters(iteration, [10.0, 10.0], is_bce=1)

    def get_version(self, iteration):
        return self.iteration_table[iteration].get_version()

    def get_l1_weight(self, iteration):
        return self.iteration_table[iteration].get_weight(0)

    def get_lpips_weight(self, iteration):
        return self.iteration_table[iteration].get_weight(1)

    def is_bce_enabled(self, iteration):
        return self.iteration_table[iteration].is_bce_enabled()


class IterationParameters():
    #6 weights total
    def __init__(self, iteration, weight_list, is_bce):
        self.iteration = iteration
        self.weight_list = weight_list
        self.is_bce = is_bce

    def get_version(self):
        return self.iteration

    def get_weight(self, index):
        if (index < len(self.weight_list)):
            return self.weight_list[index]
        else:
            # print("Weight index "+str(index)+ " not found. Returning 0.0")
            return 0.0

    def is_bce_enabled(self):
        return self.is_bce

class PairedTrainer:
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

        self.iteration = opts.iteration
        self.it_table = IterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration)
        self.use_mask = opts.use_mask
        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(self.gpu_device)
        self.l1_loss = nn.L1Loss()

        self.num_blocks = opts.num_blocks
        self.batch_size = opts.batch_size
        self.net_config = opts.net_config

        if(self.net_config == 1):
            self.G_A = cycle_gan.Generator(input_nc=3, output_nc=3, n_residual_blocks=self.num_blocks).to(self.gpu_device)
        elif(self.net_config == 2):
            self.G_A = unet_gan.UnetGenerator(input_nc=3, output_nc=3, num_downs=self.num_blocks).to(self.gpu_device)
        else:
            self.G_A = ffa.FFA(gps=3, blocks=self.num_blocks).to(self.gpu_device)

        self.D_A = cycle_gan.Discriminator().to(self.gpu_device)  # use CycleGAN's discriminator
        self.D_A_pool = image_pool.ImagePool(50)

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
        self.losses_dict[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[constants.G_LOSS_KEY] = "G loss per iteration"
        self.caption_dict[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[constants.LIKENESS_LOSS_KEY] = "Likeness loss per iteration"
        self.caption_dict[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[constants.D_A_FAKE_LOSS_KEY] = "D(A) fake loss per iteration"
        self.caption_dict[constants.D_A_REAL_LOSS_KEY] = "D(A) real loss per iteration"

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            loss = nn.L1Loss()
            return loss(pred, target)
        else:
            loss = nn.BCEWithLogitsLoss()
            return loss(pred, target)

    def calculate_lpips_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def likeness_loss(self, pred, target):
        return self.l1_loss(pred, target)

    def update_penalties(self, adv_weight):
        # what penalties to use for losses?
        self.adv_weight = adv_weight

        # save hyperparameters for bookeeping
        HYPERPARAMS_PATH = "checkpoint/" + constants.UNLIT_VERSION + "_" + constants.ITERATION + ".config"
        with open(HYPERPARAMS_PATH, "w") as f:
            print("Version: ", constants.UNLIT_TRANSFER_CHECKPATH, file=f)
            print("Learning rate for G: ", str(self.g_lr), file=f)
            print("Learning rate for D: ", str(self.d_lr), file=f)
            print("====================================", file=f)
            print("Adv weight: ", str(self.adv_weight), file=f)
            print("Likeness weight: ", str(self.it_table.get_l1_weight(self.iteration)), file=f)
            print("LPIPS weight: ", str(self.it_table.get_lpips_weight(self.iteration)), file=f)

    def train(self, a_tensor, b_tensor, train_masks):
        with amp.autocast():
            if(self.use_mask == 1):
                a_tensor = torch.mul(a_tensor, train_masks)
                b_tensor = torch.mul(b_tensor, train_masks)

            a2b = self.G_A(a_tensor)
            self.D_A.train()
            self.optimizerD.zero_grad()

            prediction = self.D_A(b_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(b_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A_pool.query(self.D_A(a2b.detach())), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.2):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            self.G_A.train()
            self.optimizerG.zero_grad()

            clean_like = self.G_A(a_tensor)
            likeness_loss = self.likeness_loss(clean_like, b_tensor) * self.it_table.get_l1_weight(self.iteration)
            lpips_loss = self.calculate_lpips_loss(clean_like, b_tensor) * self.it_table.get_lpips_weight(self.iteration)

            prediction = self.D_A(a2b)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = A_adv_loss + likeness_loss + lpips_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG)
            self.schedulerG.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict[constants.G_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict[constants.LIKENESS_LOSS_KEY].append(likeness_loss.item())
            self.losses_dict[constants.LPIP_LOSS_KEY].append(lpips_loss.item())
            self.losses_dict[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())

    def infer(self, a_tensor):
        with torch.no_grad():
            a2b = self.G_A(a_tensor)

        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, constants.UNLIT_TRANSFER_CHECKPATH)

    def visdom_visualize(self, a_tensor, b_tensor, train_masks, label = "Train"):
        with torch.no_grad():
            # report to visdom
            if (self.use_mask == 1):
                a_tensor = torch.mul(a_tensor, train_masks)
                b_tensor = torch.mul(b_tensor, train_masks)
                a2b = self.G_A(a_tensor)
            else:
                a2b = self.G_A(a_tensor)

            self.visdom_reporter.plot_image(a_tensor, str(label) + " Training A images - " + constants.UNLIT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(a2b, str(label) + " Training A2B images - " + constants.UNLIT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(b_tensor, str(label) + " B images - " + constants.UNLIT_VERSION + constants.ITERATION)

    def visdom_infer(self, rw_tensor):
        with torch.no_grad():
            rw2b = self.G_A(rw_tensor)
            self.visdom_reporter.plot_image(rw_tensor, "Real World images - " + constants.UNLIT_VERSION + constants.ITERATION)
            self.visdom_reporter.plot_image(rw2b, "Real World A2B images - " + constants.UNLIT_VERSION + constants.ITERATION)


    def load_saved_state(self, checkpoint):
        self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
        self.optimizerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY])
        self.optimizerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY])
        self.schedulerG.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler"])
        self.schedulerD.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler"])

    def save_states(self, epoch, iteration, last_metric):
        save_dict = {'epoch': epoch, 'iteration': iteration, 'net_config': self.net_config, 'num_blocks' : self.num_blocks}
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

        torch.save(save_dict, constants.UNLIT_TRANSFER_CHECKPATH)

        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_states_checkpt(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration, 'net_config': self.net_config, 'num_blocks' : self.num_blocks}
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

        torch.save(save_dict, constants.UNLIT_TRANSFER_CHECKPATH + '.checkpt')

        print("Saved model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
