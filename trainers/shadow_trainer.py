from config import iid_server_config
from trainers import abstract_iid_trainer, early_stopper
import kornia
from model.modules import image_pool
from model import vanilla_cycle_gan as cycle_gan
import constants
import torch
import torch.cuda.amp as amp
import itertools
import numpy as np
import torch.nn as nn
from hyperparam_tables import shadow_iteration_table
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
        self.it_table = shadow_iteration_table.ShadowIterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration)
        self.adv_weight = self.it_table.get_adv_weight()

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.D_SM_pool = image_pool.ImagePool(50)

        self.iid_op = iid_transforms.IIDTransform().to(self.gpu_device)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_network_config_from_version()

        self.batch_size = network_config["batch_size_z"]
        self.da_enabled = network_config["da_enabled"]

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_shadow"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shadow_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_SM_predictor.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_SM_discriminator.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.optimizerGB = torch.optim.Adam(itertools.chain(self.GB_regressor.parameters()), lr=self.g_lr)
        self.schedulerGB = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerGB, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_z_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pth'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_SM_predictor, self.D_SM_discriminator = network_creator.initialize_shadow_network(net_config, num_blocks, input_nc)

        self.GB_regressor = cycle_gan.FeatureDiscriminator(input_nc=3, output_nc=2, n_blocks=7, max_filter_size=4096, last_layer=nn.LeakyReLU).to(self.gpu_device)

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

    def masking_loss(self, pred, target):
        pred_normalized = (pred * 0.5) + 0.5
        target_normalized = (target * 0.5) + 0.5

        pred_mask = pred_normalized + (pred_normalized > 0.0).type(target_normalized.dtype)
        target_mask = target_normalized + (target_normalized > 0.0).type(target_normalized.dtype)

        # self.visdom_reporter.plot_image(pred_mask, "Pred Masks")
        # self.visdom_reporter.plot_image(target_mask, "Target Masks")

        return self.l1_loss(pred_mask, target_mask)

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict_s = {}
        self.GB_LOSS_KEY = "GB_LOSS_KEY"
        self.losses_dict_s[self.GB_LOSS_KEY] = []
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []
        self.MASK_LOSS_KEY = "MASK_LOSS_KEY"
        self.losses_dict_s[self.MASK_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[self.GB_LOSS_KEY] = "Gamma-beta loss per iteration"
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_s[self.MASK_LOSS_KEY] = "Mask loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        shadow_matte_tensor = target_map["shadow_matte"]
        gamma_beta_val = target_map["gamma_beta_val"] #TODO: Check if normalization is needed

        with amp.autocast():
            if (self.da_enabled == 1):
                input_ws = self.reshape_input(input_rgb_tensor)
            else:
                input_ws = input_rgb_tensor

            #gamme-beta regressor
            self.optimizerGB.zero_grad()
            l1_loss = self.mse_loss(self.GB_regressor(input_ws), gamma_beta_val) * self.it_table.get_gammabeta_weight()
            errGB = l1_loss

            self.fp16_scaler.scale(errGB).backward()
            self.fp16_scaler.step(self.optimizerGB)
            self.schedulerGB.step(errGB)
            self.fp16_scaler.update()

            # shadow matte discriminator
            self.optimizerD_shading.zero_grad()
            self.D_SM_discriminator.train()
            output = self.G_SM_predictor(input_ws)
            prediction = self.D_SM_discriminator(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.adversarial_loss(self.D_SM_discriminator(shadow_matte_tensor), real_tensor) * self.adv_weight
            D_SM_fake_loss = self.adversarial_loss(self.D_SM_pool.query(self.D_SM_discriminator(output.detach())), fake_tensor) * self.adv_weight

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            #shadow matte generator
            self.optimizerG_shading.zero_grad()
            self.G_SM_predictor.train()
            rgb2sm = self.G_SM_predictor(input_ws)
            SM_likeness_loss = self.l1_loss(rgb2sm, shadow_matte_tensor) * self.it_table.get_l1_weight(self.iteration)
            SM_lpip_loss = self.lpip_loss(rgb2sm, shadow_matte_tensor) * self.it_table.get_lpip_weight(self.iteration)
            SM_masking_loss = self.masking_loss(rgb2sm, shadow_matte_tensor) * self.it_table.get_masking_weight(self.iteration)
            prediction = self.D_SM_discriminator(rgb2sm)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = SM_likeness_loss + SM_lpip_loss + SM_masking_loss + SM_adv_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[self.GB_LOSS_KEY].append(errGB.item())
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(SM_lpip_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(SM_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item())
            self.losses_dict_s[self.MASK_LOSS_KEY].append(SM_masking_loss.item())

            _, rgb2sm, rgb2relit = self.test(input_map)
            self.stopper_method.register_metric(rgb2sm, shadow_matte_tensor, epoch)
            self.stop_result = self.stopper_method.test(epoch)

            if (self.stopper_method.has_reset()):
                self.save_states(epoch, iteration, False)


    def test(self, input_map):
        with torch.no_grad():
            input_rgb_tensor = input_map["rgb"]

            if (self.da_enabled == 1):
                input_ws = self.reshape_input(input_rgb_tensor)
            else:
                input_ws = input_rgb_tensor

            # print("Tensor properties. Min: ", torch.min(input_ws), " Max:", torch.max(input_ws))
            rgb2sm = self.G_SM_predictor(input_ws)
            gamma_pred, beta_pred = torch.split(self.GB_regressor(input_ws), [1, 1], 1)

            rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
            input_ws = tensor_utils.normalize_to_01(input_ws)

            rgb2relit = self.iid_op.extract_relit_batch(input_ws, gamma_pred, beta_pred)
            # rgb2relit = self.iid_op.extract_relit_batch(input_ws, torch.full_like(gamma_pred, self.iid_op.GAMMA), torch.full_like(beta_pred, self.iid_op.BETA))
            rgb2ns = self.iid_op.remove_shadow(input_ws, rgb2relit, rgb2sm)

        return rgb2ns, rgb2sm, rgb2relit

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        rgb2ns, rgb2sm, rgb2relit = self.test(input_map)
        input_ns = input_map["rgb_ns"]

        # input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
        # shadow_matte_tensor = tensor_utils.normalize_to_01(shadow_matte_tensor)
        #
        # rgb2relit = self.iid_op.extract_relit(input_rgb_tensor, self.iid_op.GAMMA, self.iid_op.BETA)
        # rgb2relit = tensor_utils.normalize_to_01(rgb2relit)
        #
        # rgb2ns = self.iid_op.remove_shadow(input_rgb_tensor, rgb2relit, shadow_matte_tensor)
        # rgb2ns = tensor_utils.normalize_to_01(rgb2ns)

        input_ns = tensor_utils.normalize_to_01(input_ns)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2relit, str(label) + " RGB Relit-Like images - " + self.NETWORK_VERSION + str(self.iteration))
        if("rgb_relit" in input_map):
            rgb_ws_relit = input_map["rgb_relit"]
            self.visdom_reporter.plot_image(rgb_ws_relit, str(label) + " RGB Relit images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2sm, str(label) + " Shadow-Like images - " + self.NETWORK_VERSION + str(self.iteration))
        if ("shadow_matte" in input_map):
            shadow_matte_tensor = input_map["shadow_matte"]
            shadow_matte_tensor = tensor_utils.normalize_to_01(shadow_matte_tensor)
            self.visdom_reporter.plot_image(shadow_matte_tensor, str(label) + " Shadow images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2ns, str(label) + " RGB-NS (Equation) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(input_ns, str(label) + " RGB No Shadow Images - " + self.NETWORK_VERSION + str(self.iteration))


    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
            print("Loaded shadow network: ", self.NETWORK_CHECKPATH)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("Loaded shadow network: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new shadow network: ", self.NETWORK_CHECKPATH)

        if(checkpoint != None):
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_shadow", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_SM_predictor.load_state_dict(checkpoint[constants.GENERATOR_KEY + "NS"])
            self.D_SM_discriminator.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "NS"])
            self.GB_regressor.load_state_dict(checkpoint[constants.GENERATOR_KEY + "GB"])

            self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.optimizerGB.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "GB"])
            self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "Z"])
            self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"])
            self.schedulerGB.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "GB"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()
        netGB_state_dict = self.GB_regressor.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        optimizerGB_state_dict = self.optimizerGB.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()
        schedulerGB_state_dict = self.schedulerGB.state_dict()

        save_dict[constants.GENERATOR_KEY + "NS"] = netGNS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "NS"] = netDNS_state_dict
        save_dict[constants.GENERATOR_KEY + "GB"] = netGB_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "GB"] = optimizerGB_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "Z"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"] = schedulerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "GB"] = schedulerGB_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))