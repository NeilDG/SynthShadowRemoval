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

class AlbedoTrainer(abstract_iid_trainer.AbstractIIDTrainer):

    def __init__(self, gpu_device, opts):
        super().__init__(gpu_device, opts)
        self.initialize_train_config(opts)

    def initialize_train_config(self, opts):
        self.iteration = opts.iteration
        self.it_table = iteration_table.IterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration, IterationTable.NetworkType.ALBEDO)
        self.adv_weight = self.it_table.get_adv_weight()
        self.rgb_l1_weight = self.it_table.get_rgb_recon_weight()

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.multiscale_grad_loss = iid_losses.MultiScaleGradientLoss(4)
        self.reflect_cons_loss = iid_losses.ReflectConsistentLoss(sample_num_per_area=1, split_areas=(1, 1))

        self.D_A_pool = image_pool.ImagePool(50)

        self.iid_op = iid_transforms.IIDTransform().to(self.gpu_device)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_network_config_from_version(opts.version)
        self.da_enabled = network_config["da_enabled"]
        self.batch_size = network_config["batch_size_a"]
        self.albedo_mode = network_config["albedo_mode"]

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_albedo"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_albedo_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.NETWORK_VERSION = sc_instance.get_version_config("network_a_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_albedo_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_A, self.D_A = network_creator.initialize_albedo_network(net_config, num_blocks, input_nc)
        self.optimizerG_albedo = torch.optim.Adam(itertools.chain(self.G_A.parameters()), lr=self.g_lr)
        self.optimizerD_albedo = torch.optim.Adam(itertools.chain(self.D_A.parameters()), lr=self.d_lr)
        self.schedulerG_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_albedo, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_albedo = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_albedo, patience=100000 / self.batch_size, threshold=0.00005)

    def initialize_dict(self):
        # what to store in visdom?
        self.GRADIENT_LOSS_KEY = "GRADIENT_LOSS_KEY"
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.MS_GRAD_LOSS_KEY = "MS_GRAD_LOSS_KEY"
        self.REFLECTIVE_LOSS_KEY = "REFLECTIVE_LOSS_KEY"

        self.losses_dict_a = {}
        self.losses_dict_a[constants.G_LOSS_KEY] = []
        self.losses_dict_a[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_a[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_a[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_a[constants.SSIM_LOSS_KEY] = []
        self.losses_dict_a[self.GRADIENT_LOSS_KEY] = []
        self.losses_dict_a[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_a[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = []
        self.losses_dict_a[self.MS_GRAD_LOSS_KEY] = []
        self.losses_dict_a[self.REFLECTIVE_LOSS_KEY] = []

        self.caption_dict_a = {}
        self.caption_dict_a[constants.G_LOSS_KEY] = "Albedo G loss per iteration"
        self.caption_dict_a[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_a[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_a[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_a[constants.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict_a[self.GRADIENT_LOSS_KEY] = "Gradient loss per iteration"
        self.caption_dict_a[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_a[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_a[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"
        self.caption_dict_a[self.MS_GRAD_LOSS_KEY] = "Multiscale gradient loss per iteration"
        self.caption_dict_a[self.REFLECTIVE_LOSS_KEY] = "Reflective loss per iteration"

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

    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        unlit_tensor = input_map["unlit"]
        shading_tensor = input_map["shading"]
        shadow_tensor = input_map["shadow"]
        albedo_tensor = target_map["albedo"]
        with amp.autocast():
            if (self.albedo_mode == 2):
                input = unlit_tensor
                # print("Using unlit tensor")
            else:
                input = input_rgb_tensor

            # albedo_masks = self.iid_op.create_sky_reflection_masks(albedo_tensor)
            # albedo_tensor = albedo_tensor * albedo_masks
            # albedo_masks = torch.cat([albedo_masks, albedo_masks, albedo_masks], 1)
            albedo_masks = torch.ones_like(albedo_tensor)

            if (self.da_enabled == 1):
                input = self.reshape_input(input)

            # produce initial albedo
            rgb2albedo = self.G_A(input)

            # albedo discriminator
            self.D_A.train()
            self.optimizerD_albedo.zero_grad()
            prediction = self.D_A(albedo_tensor)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_A_real_loss = self.adversarial_loss(self.D_A(albedo_tensor), real_tensor) * self.adv_weight
            D_A_fake_loss = self.adversarial_loss(self.D_A_pool.query(self.D_A(rgb2albedo.detach())), fake_tensor) * self.adv_weight

            errD = D_A_real_loss + D_A_fake_loss

            self.fp16_scaler.scale(errD).backward()

            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.schedulerD_albedo.step(errD)
                self.fp16_scaler.step(self.optimizerD_albedo)

            self.optimizerG_albedo.zero_grad()

            # albedo generator
            self.G_A.train()
            rgb2albedo = self.G_A(input)
            A_likeness_loss = self.l1_loss(rgb2albedo, albedo_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_lpip_loss = self.lpip_loss(rgb2albedo, albedo_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_ssim_loss = self.ssim_loss(rgb2albedo, albedo_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_gradient_loss = self.gradient_loss_term(rgb2albedo, albedo_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_ms_grad_loss = self.multiscale_grad_loss(rgb2albedo, albedo_tensor, albedo_masks.float()) * self.it_table.get_multiscale_weight(self.iteration, IterationTable.NetworkType.ALBEDO)
            A_reflective_loss = self.reflect_cons_loss(rgb2albedo, albedo_tensor, input_rgb_tensor, albedo_masks.float()) * self.it_table.get_reflect_cons_weight(self.iteration, IterationTable.NetworkType.ALBEDO)

            prediction = self.D_A(rgb2albedo)
            real_tensor = torch.ones_like(prediction)
            A_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            rgb_like = self.iid_op.produce_rgb(rgb2albedo, shading_tensor, shadow_tensor)
            rgb_l1_loss = self.l1_loss(rgb_like, input_rgb_tensor) * self.rgb_l1_weight


            errG = A_likeness_loss + A_lpip_loss + A_ssim_loss + A_gradient_loss + A_adv_loss + A_ms_grad_loss + A_reflective_loss + rgb_l1_loss
            self.fp16_scaler.scale(errG).backward()
            self.schedulerG_albedo.step(errG)
            self.fp16_scaler.step(self.optimizerG_albedo)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_a[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_a[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_a[constants.LIKENESS_LOSS_KEY].append(A_likeness_loss.item())
            self.losses_dict_a[constants.LPIP_LOSS_KEY].append(A_lpip_loss.item())
            self.losses_dict_a[constants.SSIM_LOSS_KEY].append(A_ssim_loss.item())
            self.losses_dict_a[self.GRADIENT_LOSS_KEY].append(A_gradient_loss.item())
            self.losses_dict_a[constants.G_ADV_LOSS_KEY].append(A_adv_loss.item())
            self.losses_dict_a[constants.D_A_FAKE_LOSS_KEY].append(D_A_fake_loss.item())
            self.losses_dict_a[constants.D_A_REAL_LOSS_KEY].append(D_A_real_loss.item())
            self.losses_dict_a[self.RGB_RECONSTRUCTION_LOSS_KEY].append(rgb_l1_loss.item())
            self.losses_dict_a[self.MS_GRAD_LOSS_KEY].append(A_ms_grad_loss.item())
            self.losses_dict_a[self.REFLECTIVE_LOSS_KEY].append(A_reflective_loss.item())

            self.stopper_method.register_metric(self.test(input_map), albedo_tensor, epoch)
            self.stop_result = self.stopper_method.test(epoch)

            if (self.stopper_method.has_reset()):
                self.save_states(epoch, iteration, False)

    def is_stop_condition_met(self):
        return self.stop_result

    def test(self, input_map):
        with torch.no_grad():
            input_rgb_tensor = input_map["rgb"]

            if (self.albedo_mode == 2):
                unlit_tensor = input_map["unlit"]
                input = unlit_tensor
            else:
                input = input_rgb_tensor

            if (self.da_enabled == 1):
                input = self.reshape_input(input)

            rgb2albedo = self.G_A(input)
        return rgb2albedo

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_a, self.caption_dict_a, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        albedo_tensor = input_map["albedo"]
        unlit_tensor = input_map["unlit"]
        shading_tensor = input_map["shading"]
        shadow_tensor = input_map["shadow"]

        # mask_tensor = self.iid_op.create_sky_reflection_masks(albedo_tensor)
        embedding_rep = self.get_feature_rep(input_rgb_tensor)
        rgb2albedo = self.test(input_map)

        # normalize to 0-1
        input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
        rgb2albedo = tensor_utils.normalize_to_01(rgb2albedo)

        # rgb2albedo = rgb2albedo * mask_tensor
        # rgb2albedo = self.iid_op.mask_fill_nonzeros(rgb2albedo)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(unlit_tensor, str(label) + " Unlit Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(self.iid_op.produce_rgb(rgb2albedo, shading_tensor, shadow_tensor), str(label) + " RGB Reconstructed Images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2albedo, str(label) + " RGB2Albedo images - " + self.NETWORK_VERSION + str(self.iteration), True)
        self.visdom_reporter.plot_image(albedo_tensor, str(label) + " Albedo images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(shading_tensor, str(label) + " Shading images - " + self.NETWORK_VERSION + str(self.iteration))

    def visdom_infer(self, input_map):
        input_rgb_tensor = input_map["rgb"]
        embedding_rep = self.get_feature_rep(input_rgb_tensor)
        rgb2albedo = self.test(input_map)

        self.visdom_reporter.plot_image(input_rgb_tensor, "Real World images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(embedding_rep, "Real World Embeddings - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2albedo, "Real World A2B images - " + self.NETWORK_VERSION + str(self.iteration))

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
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_albedo", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_A.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
            self.D_A.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "A"])
            self.optimizerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"])
            self.optimizerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"])
            self.schedulerG_albedo.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "A"])
            self.schedulerD_albedo.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "A"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGA_state_dict = self.G_A.state_dict()
        netDA_state_dict = self.D_A.state_dict()
        optimizerGalbedo_state_dict = self.optimizerG_albedo.state_dict()
        optimizerDalbedo_state_dict = self.optimizerD_albedo.state_dict()
        schedulerGalbedo_state_dict = self.schedulerG_albedo.state_dict()
        schedulerDalbedo_state_dict = self.schedulerD_albedo.state_dict()
        save_dict[constants.GENERATOR_KEY + "A"] = netGA_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "A"] = netDA_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "A"] = optimizerDalbedo_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "A"] = schedulerGalbedo_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "A"] = schedulerDalbedo_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
