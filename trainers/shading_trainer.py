from config import iid_server_config
from trainers import abstract_iid_trainer, early_stopper
import kornia
from model import iteration_table, embedding_network
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from model import usi3d_gan
from model.modules import image_pool
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from model.iteration_table import IterationTable
from trainers import paired_trainer
from transforms import iid_transforms
from utils import plot_utils
from utils import tensor_utils
from custom_losses import ssim_loss, iid_losses
import lpips

class ShadingTrainer(abstract_iid_trainer.AbstractIIDTrainer):

    def __init__(self, gpu_device, opts):
        super().__init__(gpu_device, opts)
        self.initialize_train_config(opts)

    def initialize_train_config(self, opts):
        self.iteration = opts.iteration
        self.it_table = iteration_table.IterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration, IterationTable.NetworkType.SHADING)
        self.adv_weight = self.it_table.get_adv_weight()
        self.rgb_l1_weight = self.it_table.get_rgb_recon_weight()

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multiscale_grad_loss = iid_losses.MultiScaleGradientLoss(4)
        self.reflect_cons_loss = iid_losses.ReflectConsistentLoss(sample_num_per_area=1, split_areas=(1, 1))

        self.D_S_pool = image_pool.ImagePool(50)
        self.D_Z_pool = image_pool.ImagePool(50)

        self.iid_op = iid_transforms.IIDTransform().to(self.gpu_device)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_network_config_from_version()
        self.batch_size = network_config["batch_size_s"]
        self.da_enabled = network_config["da_enabled"]

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_shading"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shading_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_S.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_S.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_s_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shading_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_S, self.D_S = network_creator.initialize_shading_network(net_config, num_blocks, input_nc)

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.mse_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

    def gradient_loss_term(self, pred, target):
        pred_gradient = kornia.filters.spatial_gradient(pred)
        target_gradient = kornia.filters.spatial_gradient(target)

        return self.mse_loss(pred_gradient, target_gradient)

    def lpip_loss(self, pred, target):
        result = torch.squeeze(self.lpips_loss(pred, target))
        result = torch.mean(result)
        return result

    def ssim_loss(self, pred, target):
        pred_normalized = (pred * 0.5) + 0.5
        target_normalized = (target * 0.5) + 0.5

        return self.ssim_loss(pred_normalized, target_normalized)

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_s[constants.SSIM_LOSS_KEY] = []
        self.GRADIENT_LOSS_KEY = "GRADIENT_LOSS_KEY"
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.losses_dict_s[self.GRADIENT_LOSS_KEY] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shading G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_s[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        albedo_tensor = input_map["albedo"]
        shading_tensor = target_map["shading"]
        shadow_tensor = target_map["shadow"]

        with amp.autocast():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)
            else:
                input = input_rgb_tensor

            self.optimizerD_shading.zero_grad()

            #shading discriminator
            rgb2shading = self.G_S(input)
            self.D_S.train()
            prediction = self.D_S(shading_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_S_real_loss = self.adversarial_loss(self.D_S(shading_tensor), real_tensor) * self.adv_weight
            D_S_fake_loss = self.adversarial_loss(self.D_S_pool.query(self.D_S(rgb2shading.detach())), fake_tensor) * self.adv_weight

            errD = D_S_real_loss + D_S_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            #shading generator
            self.G_S.train()
            rgb2shading = self.G_S(input)
            S_likeness_loss = self.l1_loss(rgb2shading, shading_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_lpip_loss = self.lpip_loss(rgb2shading, shading_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_ssim_loss = self.ssim_loss(rgb2shading, shading_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADING)
            S_gradient_loss = self.gradient_loss_term(rgb2shading, shading_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADING)
            prediction = self.D_S(rgb2shading)
            real_tensor = torch.ones_like(prediction)
            S_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = self.iid_op.produce_rgb(albedo_tensor, self.G_S(input_rgb_tensor), shadow_tensor)
            rgb_l1_loss = self.l1_loss(rgb_like, input_rgb_tensor) * self.rgb_l1_weight

            errG = S_likeness_loss + S_lpip_loss + S_ssim_loss + S_gradient_loss + S_adv_loss + rgb_l1_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(S_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(S_lpip_loss.item())
            self.losses_dict_s[constants.SSIM_LOSS_KEY].append(S_ssim_loss.item())
            self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(S_gradient_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(S_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_S_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_S_real_loss.item())
            self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

            rgb2shading = self.test(input_map)
            self.stopper_method.register_metric(rgb2shading, shading_tensor, epoch)
            self.stop_result = self.stopper_method.test(epoch)

            if (self.stopper_method.has_reset()):
                self.save_states(epoch, iteration, False)

    def test(self, input_map):
        with torch.no_grad():
            input_rgb_tensor = input_map["rgb"]
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)
            else:
                input = input_rgb_tensor

            rgb2shading = self.G_S(input)
        return rgb2shading

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        shading_tensor = input_map["shading"]

        embedding_rep = self.get_feature_rep(input_rgb_tensor)
        rgb2shading = self.test(input_map)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2shading, str(label) + " RGB2Shading images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + self.NETWORK_VERSION + str(self.iteration))

    def visdom_infer(self, input_map):
        input_rgb_tensor = input_map["rgb"]
        rgb2shading = self.test(input_map)

        self.visdom_reporter.plot_image(input_rgb_tensor, "Real World images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2shading, "Real WorldRGB2Shading images - " + self.NETWORK_VERSION + str(self.iteration))

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

        if(checkpoint != None):
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_shading", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_S.load_state_dict(checkpoint[constants.GENERATOR_KEY + "S"])
            self.D_S.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "S"])
            self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"])
            self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"])
            self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "S"])
            self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "S"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGS_state_dict = self.G_S.state_dict()
        netDS_state_dict = self.D_S.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()

        save_dict[constants.GENERATOR_KEY + "S"] = netGS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "S"] = netDS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "S"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "S"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "S"] = schedulerDshading_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))