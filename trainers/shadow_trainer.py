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

class ShadowTrainer(abstract_iid_trainer.AbstractIIDTrainer):

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
        network_config = sc_instance.interpret_network_config_from_version(opts.version)
        self.batch_size = network_config["batch_size_s"]
        self.da_enabled = network_config["da_enabled"]

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_shading"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shadow_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_Z.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_Z.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_Z_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_Z, self.D_Z = network_creator.initialize_shadow_network(net_config, num_blocks, input_nc)

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
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shadow G loss per iteration"
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
        input_rgb_tensor_noshadow = input_map["rgb_ns"]
        shadow_tensor = target_map["shadow"]

        with amp.autocast():
            if (self.da_enabled == 1):
                input = self.reshape_input(input_rgb_tensor)
            else:
                input = input_rgb_tensor

            self.optimizerD_shading.zero_grad()

            # shadow discriminator
            rgb2shadow = self.G_Z(input)
            self.D_Z.train()
            prediction = self.D_Z(shadow_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_Z_real_loss = self.adversarial_loss(self.D_Z(shadow_tensor), real_tensor) * self.adv_weight
            D_Z_fake_loss = self.adversarial_loss(self.D_Z_pool.query(self.D_Z(rgb2shadow.detach())), fake_tensor) * self.adv_weight

            errD = D_Z_real_loss + D_Z_fake_loss
            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            # shadow generator
            self.G_Z.train()
            rgb2shadow = self.G_Z(input)
            Z_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            Z_gradient_loss = self.gradient_loss_term(rgb2shadow, shadow_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            prediction = self.D_Z(rgb2shadow)
            real_tensor = torch.ones_like(prediction)
            Z_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_noshadow_like = self.iid_op.remove_rgb_shadow(input_rgb_tensor, self.G_Z(input_rgb_tensor))
            rgb_l1_loss = self.l1_loss(rgb_noshadow_like, input_rgb_tensor_noshadow) * self.rgb_l1_weight

            errG = Z_likeness_loss + Z_lpip_loss + Z_ssim_loss + Z_gradient_loss + Z_adv_loss + rgb_l1_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(Z_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(Z_lpip_loss.item())
            self.losses_dict_s[constants.SSIM_LOSS_KEY].append(Z_ssim_loss.item())
            self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(Z_gradient_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(Z_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_Z_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_Z_real_loss.item())
            self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())

            rgb2shadow = self.test(input_map)
            self.stopper_method.register_metric(rgb2shadow, shadow_tensor, epoch)
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

            rgb2shadow = self.G_Z(input)

        return rgb2shadow

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        shadow_tensor = input_map["shadow"]

        embedding_rep = self.get_feature_rep(input_rgb_tensor)
        rgb2shadow = self.test(input_map)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2shadow, str(label) + " RGB2Shadow images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + self.NETWORK_VERSION + str(self.iteration))