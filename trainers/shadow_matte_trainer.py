from pathlib import Path

import torchvision.transforms.functional

from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer, early_stopper
import kornia
from model.modules import image_pool
from model import vanilla_cycle_gan as cycle_gan
import global_config
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from hyperparam_tables import shadow_iteration_table
from transforms import iid_transforms, shadow_map_transforms
from utils import plot_utils
from utils import tensor_utils
from losses import common_losses
from model.modules import shadow_matte_pool
from trainers import best_tracker

class ShadowMatteTrainer(abstract_iid_trainer.AbstractIIDTrainer):
    def __init__(self, gpu_device):
        super().__init__(gpu_device)
        self.initialize_train_config()

    def initialize_train_config(self):
        self.iteration = global_config.sm_iteration
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()

        self.D_SM_pool = image_pool.ImagePool(50)
        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision
        self.common_losses = common_losses.LossRepository(self.gpu_device, self.iteration)

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.load_size = global_config.load_size
        self.batch_size = global_config.batch_size
        self.use_istd_pool = config_holder.get_network_attribute("use_istd_pool", False)

        if(self.use_istd_pool):
            shadow_matte_pool.ShadowMattePool.initialize()
            self.ISTD_SM_pool = shadow_matte_pool.ShadowMattePool().getInstance()

        self.stopper_method = early_stopper.EarlyStopper(network_config["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, 3000, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shadow_network()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_SM_predictor.parameters()), lr=network_config["g_lr"], weight_decay=network_config["weight_decay"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_SM_discriminator.parameters()), lr=network_config["d_lr"], weight_decay=network_config["weight_decay"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = config_holder.get_sm_version_name()
        self.BEST_NETWORK_SAVE_PATH = "./checkpoint/best/"
        if (global_config.load_per_epoch == False and global_config.load_per_sample == False):
            if (global_config.load_best):
                self.NETWORK_CHECKPATH = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + '_best.pth'
            else:
                self.NETWORK_CHECKPATH = './checkpoint/' + self.NETWORK_VERSION + '.pt'
            self.load_last_epoch_val()
            self.load_saved_state()
        elif (global_config.load_per_epoch == True):
            self.NETWORK_SAVE_PATH = "./checkpoint/by_epoch/"
        else:
            self.NETWORK_SAVE_PATH = "./checkpoint/by_sample/"

        self.BEST_NETWORK_SAVE_PATH = "./checkpoint/best/"
        network_file_name = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_best" + ".pth"
        self.best_tracker = best_tracker.BestTracker(early_stopper.EarlyStopperMethod.L1_TYPE)
        self.best_tracker.load_best_state(network_file_name)

        # model_parameters = filter(lambda p: p.requires_grad, self.G_SM_predictor.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print("-----------------------")
        # print("DSP-FFANet Shadow Matte total parameters: ", params)
        # print("-----------------------")

    def initialize_shadow_network(self):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_SM_predictor, self.D_SM_discriminator = network_creator.initialize_shadow_matte_network()  # shadow map (Shadow - Shadow-Free)

    def initialize_dict(self):
        self.G_LOSS_KEY = "g_loss"
        self.IDENTITY_LOSS_KEY = "id"
        self.CYCLE_LOSS_KEY = "cyc"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.LIKENESS_LOSS_KEY = "likeness"
        self.LPIP_LOSS_KEY = "lpip"
        self.SMOOTHNESS_LOSS_KEY = "smoothness"
        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_A_REAL_LOSS_KEY = "d_real_a"
        self.D_A_FAKE_LOSS_KEY = "d_fake_a"
        self.D_B_REAL_LOSS_KEY = "d_real_b"
        self.D_B_FAKE_LOSS_KEY = "d_fake_b"
        self.MASK_LOSS_KEY = "MASK_LOSS_KEY"
        self.ISTD_SM_LOSS_KEY = "ISTD_SM_LOSS_KEY"

        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[self.G_LOSS_KEY] = []
        self.losses_dict_s[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[self.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[self.LPIP_LOSS_KEY] = []
        self.losses_dict_s[self.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[self.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[self.D_A_REAL_LOSS_KEY] = []
        self.losses_dict_s[self.MASK_LOSS_KEY] = []
        self.losses_dict_s[self.ISTD_SM_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict_s[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[self.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[self.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[self.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[self.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_s[self.MASK_LOSS_KEY] = "Mask loss per iteration"
        self.caption_dict_s[self.ISTD_SM_LOSS_KEY] = "ISTD SM loss per iterationm"

        # what to store in visdom?
        self.losses_dict_t = {}

        self.TRAIN_LOSS_KEY = "TRAIN_LOSS_KEY"
        self.losses_dict_t[self.TRAIN_LOSS_KEY] = []
        self.TEST_LOSS_KEY = "TEST_LOSS_KEY"
        self.losses_dict_t[self.TEST_LOSS_KEY] = []

        self.caption_dict_t = {}
        self.caption_dict_t[self.TRAIN_LOSS_KEY] = "Train L1 loss per iteration"
        self.caption_dict_t[self.TEST_LOSS_KEY] = "Test L1 loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        input_ws = input_map["rgb"]
        target_tensor = target_map["shadow_matte"]
        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            # shadow map discriminator
            self.optimizerD.zero_grad()
            self.D_SM_discriminator.train()
            output = self.G_SM_predictor(input_ws)
            prediction = self.D_SM_discriminator(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.common_losses.compute_adversarial_loss(self.D_SM_discriminator(target_tensor), real_tensor)
            D_SM_fake_loss = self.common_losses.compute_adversarial_loss(self.D_SM_pool.query(self.D_SM_discriminator(output.detach())), fake_tensor)

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            #shadow map generator
            self.optimizerG.zero_grad()
            self.G_SM_predictor.train()
            rgb2sm = self.G_SM_predictor(input_ws)
            SM_likeness_loss = self.common_losses.compute_l1_loss(rgb2sm, target_tensor)
            # SM_lpip_loss = self.lpip_loss(rgb2sm, target_tensor) * self.it_table.get_lpip_weight(self.iteration)
            # SM_masking_loss = self.masking_loss(rgb2sm, target_tensor) * self.it_table.get_masking_weight(self.iteration)
            prediction = self.D_SM_discriminator(rgb2sm)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            errG = SM_likeness_loss + SM_adv_loss

            self.fp16_scaler.scale(errG).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerG)
                self.schedulerG.step(errG)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                if (iteration > 50):
                    # what to put to losses dict for visdom reporting?
                    self.losses_dict_s[self.G_LOSS_KEY].append(errG.item())
                    self.losses_dict_s[self.D_OVERALL_LOSS_KEY].append(errD.item())
                    self.losses_dict_s[self.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
                    # self.losses_dict_s[self.LPIP_LOSS_KEY].append(SM_lpip_loss.item())
                    self.losses_dict_s[self.G_ADV_LOSS_KEY].append(SM_adv_loss.item())
                    self.losses_dict_s[self.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
                    self.losses_dict_s[self.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item())
                    # self.losses_dict_s[self.MASK_LOSS_KEY].append(SM_masking_loss.item())
                    # self.losses_dict_s[self.ISTD_SM_LOSS_KEY].append(SM_istd_loss.item())

                #perform validation test
                rgb2sm_istd = self.validation_test(input_map)
                istd_sm_test = input_map["matte_val"]

                #check and save best state
                self.try_save_best_state(rgb2sm_istd, istd_sm_test, epoch, iteration)

                #perform early stopping
                # if (global_config.save_every_epoch == False):
                #     self.stopper_method.register_metric(rgb2sm_istd, istd_sm_test, epoch)
                #     self.stop_result = self.stopper_method.test(epoch)
                #
                #     if (self.stopper_method.has_reset()):
                #         self.save_states(epoch, iteration, False)

                #plot train-test loss
                rgb2sm = self.test(input_map)
                rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
                target_tensor = tensor_utils.normalize_to_01(target_tensor)
                rgb2sm_istd = tensor_utils.normalize_to_01(rgb2sm_istd)
                istd_sm_test = tensor_utils.normalize_to_01(istd_sm_test)
                self.losses_dict_t[self.TRAIN_LOSS_KEY].append(self.common_losses.compute_l1_loss(rgb2sm, target_tensor).item())
                self.losses_dict_t[self.TEST_LOSS_KEY].append(self.common_losses.compute_l1_loss(rgb2sm_istd, istd_sm_test).item())

    def is_stop_condition_met(self):
        return self.stop_result

    def validation_test(self, input_map):
        # print("Testing on ISTD dataset.")

        input_map_new = {"rgb": input_map["rgb_ws_val"],
                         "shadow_matte": input_map["matte_val"]}

        return self.test(input_map_new)

    def test(self, input_map):
        with torch.no_grad():
            self.G_SM_predictor.eval()
            input_ws = input_map["rgb"]

            rgb2sm = self.G_SM_predictor(input_ws)
            # rgb2sm = 1.0 - rgb2sm

        return rgb2sm

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_train_test_loss("train_test_loss", iteration, self.losses_dict_t, self.caption_dict_t, self.NETWORK_VERSION + str(self.iteration))

    def visdom_visualize(self, input_map, label="Train"):
        input_ws = input_map["rgb"]

        matte_tensor = input_map["shadow_matte"]
        rgb_ns = input_map["rgb_ns"]
        rgb2sm = self.test(input_map)

        self.visdom_reporter.plot_image(input_ws, str(label) + " RGB (With Shadows) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb_ns, str(label) + " RGB (No Shadows) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(matte_tensor, str(label) + " Shadow Matte Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2sm, str(label) + " Shadow Matte-Like images - " + self.NETWORK_VERSION + str(self.iteration))

        if(self.use_istd_pool):
            istd_matte_tensors = self.ISTD_SM_pool.query_samples(16)
            self.visdom_reporter.plot_image(istd_matte_tensors, str(label) + " ISTD SM Pool - " + self.NETWORK_VERSION + str(self.iteration))

    def load_last_epoch_val(self):
        try:
            checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
            checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
        except:
            checkpoint = None
            print("No existing checkpoint file found. Creating new shadow network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            global_config.last_epoch_sm = checkpoint["epoch"]

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("No available .pt/pth file found. Loading .checkpt file: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new shadow network: ", self.NETWORK_CHECKPATH)

        if(checkpoint != None):
            # global_config.last_epoch_sm = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])
            self.G_SM_predictor.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "M"])
            self.D_SM_discriminator.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "M"])

            print("Loaded shadow matte network: ", self.NETWORK_CHECKPATH, "Epoch: ", checkpoint["epoch"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()

        save_dict[global_config.GENERATOR_KEY + "M"] = netGNS_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "M"] = netDNS_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def save_for_each_epoch(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()

        save_dict[global_config.GENERATOR_KEY + "M"] = netGNS_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "M"] = netDNS_state_dict

        network_file_name = self.NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_" + str(epoch) + ".pth"
        torch.save(save_dict, network_file_name)
        print("Saved stable model state. Epoch: %d. Name: %s" % (epoch, network_file_name))

    def load_specific_epoch(self, epoch):
        network_file_name = self.NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_" + str(epoch) + ".pth"
        try:
            checkpoint = torch.load(network_file_name, map_location=self.gpu_device)
        except:
            checkpoint = None
            print("No existing checkpoint file found: ", network_file_name)

        if (checkpoint != None):
            global_config.last_epoch_sm = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])
            self.G_SM_predictor.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "Z"])
            self.D_SM_discriminator.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "Z"])

            print("Loaded shadow matte network: ", network_file_name, "Epoch: ", checkpoint["epoch"])

    def save_state_for_sample(self, epoch, iteration):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()

        save_dict[global_config.GENERATOR_KEY + "M"] = netGNS_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "M"] = netDNS_state_dict

        network_file_name = self.NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_" + str(global_config.img_to_load) + ".pth"
        torch.save(save_dict, network_file_name)
        print("Saved stable model state: %s Epoch: %d. Name: %s" % (len(save_dict), (epoch), network_file_name))

    def load_state_for_specific_sample(self):
        network_file_name = self.NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_" + str(global_config.img_to_load) + ".pth"
        try:
            checkpoint = torch.load(network_file_name, map_location=self.gpu_device)
        except:
            checkpoint = None
            print("No existing checkpoint file found: ", network_file_name)

        if (checkpoint != None):
            global_config.last_epoch_ns = checkpoint["epoch"]
            global_config.last_iteration_ns = checkpoint["iteration"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])
            self.G_SM_predictor.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "Z"])
            self.D_SM_discriminator.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "Z"])

            print("Loaded shadow matte network: ", network_file_name, "Epoch: ", checkpoint["epoch"])

    def try_save_best_state(self, input, target, epoch, iteration):
        best_achieved = self.best_tracker.test(input, target)
        best_metric = self.best_tracker.get_best_metric()
        if(best_achieved):
            network_file_name = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_best" + ".pth"
            save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric(),
                         'best_metric' : best_metric}
            netGNS_state_dict = self.G_SM_predictor.state_dict()
            netDNS_state_dict = self.D_SM_discriminator.state_dict()

            save_dict[global_config.GENERATOR_KEY + "M"] = netGNS_state_dict
            save_dict[global_config.DISCRIMINATOR_KEY + "M"] = netDNS_state_dict

            torch.save(save_dict, network_file_name)
            print("Saved best model state. Epoch: %d. Name: %s. Best metric: %f" % (epoch, network_file_name, best_metric))













