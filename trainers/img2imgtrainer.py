import kornia.losses

from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import early_stopper
from losses import common_losses
from loaders import transform_operations

class Img2ImgTrainer(abstract_iid_trainer.AbstractIIDTrainer):
    def __init__(self, gpu_device):
        super().__init__(gpu_device)
        self.initialize_train_config()

    def initialize_train_config(self):
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()
        self.iteration = global_config.st_iteration
        self.common_losses = common_losses.LossRepository(self.gpu_device, self.iteration)
        self.l1_loss = nn.L1Loss()

        self.D_A_pool = image_pool.ImagePool(50)
        self.D_B_pool = image_pool.ImagePool(50)

        self.fp16_scaler = amp.GradScaler()
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.load_size = global_config.load_size
        self.batch_size = global_config.batch_size

        self.stopper_method = early_stopper.EarlyStopper(network_config["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, 1000)
        self.stop_result = False

        self.initialize_dict()
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_A2B, self.D_B = network_creator.initialize_img2img_network()
        self.G_B2A, self.D_A = network_creator.initialize_img2img_network()

        patch_size = config_holder.get_network_attribute("patch_size", 32)
        self.transform_op = transform_operations.Img2ImgBasicTransform(patch_size).to(self.gpu_device)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), lr=network_config["g_lr"], weight_decay=network_config["weight_decay"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=network_config["d_lr"], weight_decay=network_config["weight_decay"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = ConfigHolder.getInstance().get_st_version_name()
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_dict(self):
        # dictionary keys
        self.G_LOSS_KEY = "g_loss"
        self.IDENTITY_LOSS_KEY = "id"
        self.CYCLE_LOSS_KEY = "cyc"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.LIKENESS_LOSS_KEY = "likeness"
        self.RMSE_LOSS_KEY = "rmse_loss"
        self.SSIM_LOSS_KEY = "ssim_loss"

        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_A_LOSS_KEY = "d_a"
        self.D_B_LOSS_KEY = "d_b"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.G_LOSS_KEY] = []
        self.losses_dict[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[self.LIKENESS_LOSS_KEY] = []
        self.losses_dict[self.CYCLE_LOSS_KEY] = []
        self.losses_dict[self.IDENTITY_LOSS_KEY] = []
        self.losses_dict[self.G_ADV_LOSS_KEY] = []
        self.losses_dict[self.RMSE_LOSS_KEY] = []
        self.losses_dict[self.SSIM_LOSS_KEY] = []
        self.losses_dict[self.D_A_LOSS_KEY] = []
        self.losses_dict[self.D_B_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[self.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[self.CYCLE_LOSS_KEY] = "Cycle loss per iteration"
        self.caption_dict[self.IDENTITY_LOSS_KEY] = "Identity loss per iteration"
        self.caption_dict[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[self.RMSE_LOSS_KEY] = "RMSE loss per iteration"
        self.caption_dict[self.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[self.D_A_LOSS_KEY] = "D(A) loss per iteration"
        self.caption_dict[self.D_B_LOSS_KEY] = "D(B) real loss per iteration"

        # what to store in visdom?
        self.losses_dict_t = {}

        self.TRAIN_LOSS_KEY = "TRAIN_LOSS_KEY"
        self.losses_dict_t[self.TRAIN_LOSS_KEY] = []
        self.TEST_LOSS_KEY = "TEST_LOSS_KEY"
        self.losses_dict_t[self.TEST_LOSS_KEY] = []

        self.caption_dict_t = {}
        self.caption_dict_t[self.TRAIN_LOSS_KEY] = "Train L1 loss per iteration"
        self.caption_dict_t[self.TEST_LOSS_KEY] = "Test L1 loss per iteration"

    def compute_identity_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "id_weight")
        if (weight > 0.0):
            return self.l1_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_cycle_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "cycle_weight")
        if (weight > 0.0):
            return self.l1_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def train(self, epoch, iteration, input_map, target_map):
        img_a = input_map["img_a"]
        img_b = input_map["img_b"]
        img_a = self.transform_op(img_a)
        img_b = self.transform_op(img_b)

        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            #discriminator
            self.optimizerD.zero_grad()
            self.D_A.train()
            self.D_B.train()

            output = self.G_A2B(img_a)
            prediction = self.D_B(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.common_losses.compute_adversarial_loss(self.D_B(img_b), real_tensor)
            D_B_fake_loss = self.common_losses.compute_adversarial_loss(self.D_B_pool.query(self.D_B(output.detach())), fake_tensor)

            output = self.G_B2A(img_b)
            prediction = self.D_A(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.common_losses.compute_adversarial_loss(self.D_A(img_a), real_tensor)
            D_A_fake_loss = self.common_losses.compute_adversarial_loss(self.D_A_pool.query(self.D_A(output.detach())), fake_tensor)

            errD = D_B_real_loss + D_B_fake_loss + D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.schedulerD.step(errD)
                self.fp16_scaler.step(self.optimizerD)

            self.optimizerG.zero_grad()
            self.G_A2B.train()
            self.G_B2A.train()

            identity_b = self.G_A2B(img_b)
            img_a2b = self.G_A2B(img_a)

            B_identity_loss = self.compute_identity_loss(identity_b, img_b)
            B_likeness_loss = self.common_losses.compute_l1_loss(img_a2b, img_b)
            B_cycle_loss = self.compute_cycle_loss(self.G_A2B(self.G_B2A(img_b)), img_b)

            prediction = self.D_B(img_a2b)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            identity_a = self.G_B2A(img_a)
            img_b2a = self.G_B2A(img_b)

            A_identity_loss = self.compute_identity_loss(identity_a, img_a)
            A_likeness_loss = self.common_losses.compute_l1_loss(img_b2a, img_a)
            A_cycle_loss = self.compute_cycle_loss(self.G_B2A(self.G_A2B(img_a)), img_a)

            prediction = self.D_A(img_b2a)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            errG = A_identity_loss + B_identity_loss + A_likeness_loss + B_likeness_loss + A_adv_loss + B_adv_loss + A_cycle_loss + B_cycle_loss
            self.fp16_scaler.scale(errG).backward()

            if (accum_batch_size % self.batch_size == 0):
                self.schedulerG.step(errG)
                self.fp16_scaler.step(self.optimizerG)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                if (iteration > 10):
                    self.losses_dict[self.G_LOSS_KEY].append(errG.item())
                    self.losses_dict[self.D_OVERALL_LOSS_KEY].append(errD.item())
                    self.losses_dict[self.IDENTITY_LOSS_KEY].append(A_identity_loss.item() + B_identity_loss.item())
                    self.losses_dict[self.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
                    self.losses_dict[self.G_ADV_LOSS_KEY].append(A_adv_loss.item() + B_adv_loss.item())
                    self.losses_dict[self.D_A_LOSS_KEY].append(D_A_fake_loss.item() + D_A_real_loss.item())
                    self.losses_dict[self.D_B_LOSS_KEY].append(D_B_fake_loss.item() + D_B_real_loss.item())
                    self.losses_dict[self.CYCLE_LOSS_KEY].append(A_cycle_loss.item() + B_cycle_loss.item())

            a2b, b2a = self.test(input_map, "Train")
            self.stopper_method.register_metric(a2b, img_b, epoch)
            self.stopper_method.register_metric(b2a, img_a, epoch)
            self.stop_result = self.stopper_method.test(epoch)

            if (self.stopper_method.has_reset()):
                self.save_states(epoch, iteration, False)

    def is_stop_condition_met(self):
        return self.stopper_method.did_stop_condition_met()

    def test(self, input_map, label="Test"):
        with torch.no_grad():
            img_a = input_map["img_a"]
            img_b = input_map["img_b"]

            if(label == "Train"):
                img_a = self.transform_op(img_a)
                img_b = self.transform_op(img_b)

            self.G_A2B.eval()
            self.G_B2A.eval()
            img_a2b = self.G_A2B(img_a)
            img_b2a = self.G_B2A(img_b)
            return img_a2b, img_b2a

    def visdom_plot(self, iteration):
        style_transfer_version = global_config.style_transfer_version
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, style_transfer_version)

    def visdom_visualize(self, input_map, label="Train"):
        with torch.no_grad():
            style_transfer_version = global_config.style_transfer_version
            img_a = input_map["img_a"]
            img_b = input_map["img_b"]
            if(label == "Train"):
                img_a = self.transform_op(img_a)
                img_b = self.transform_op(img_b)

            img_a2b, img_b2a = self.test(input_map)
            img_a2b2a = self.G_B2A(self.G_A2B(img_a))
            img_b2a2b = self.G_A2B(self.G_B2A(img_b))

            self.visdom_reporter.plot_image(img_a, str(label) + " Input A Images - " + style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(img_a2b2a, str(label) + " Input A Cycle - " + style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(img_a2b, str(label) + " A2B Transfer " + style_transfer_version + str(self.iteration))

            self.visdom_reporter.plot_image(img_b, str(label) + " Input B Images - " + style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(img_b2a2b, str(label) + " Input B Cycle - " + style_transfer_version + str(self.iteration))
            self.visdom_reporter.plot_image(img_b2a, str(label) + " B2A Transfer - " + style_transfer_version + str(self.iteration))

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGA2B_state_dict = self.G_A2B.state_dict()
        netGB2A_state_dict = self.G_B2A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        save_dict[global_config.GENERATOR_KEY + "A2B"] = netGA2B_state_dict
        save_dict[global_config.GENERATOR_KEY + "B2A"] = netGB2A_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "B"] = netDB_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new depth network: ", self.NETWORK_CHECKPATH)

        if(checkpoint != None):
            global_config.last_epoch_st = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])

            self.G_A2B.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "A2B"])
            self.G_B2A.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "B2A"])
            self.D_A.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "A"])
            self.D_B.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "B"])

            print("Loaded style transfer network: ", self.NETWORK_CHECKPATH, "Epoch: ", global_config.last_epoch_st)



