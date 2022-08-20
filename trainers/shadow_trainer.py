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
        self.rgb_l1_weight = self.it_table.get_rgb_recon_weight()

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.D_SM_pool = image_pool.ImagePool(50)
        self.D_rgb_pool = image_pool.ImagePool(50)

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

        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_SM_predictor.parameters(), self.G_rgb.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_SM_discriminator.parameters(), self.D_rgb.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_z_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_SM_predictor, self.D_SM_discriminator = network_creator.initialize_shadow_network(net_config, num_blocks, input_nc)
        self.G_rgb, _ = network_creator.initialize_rgb_network(net_config, num_blocks, input_nc)
        self.D_rgb = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

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
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.MASK_LOSS_KEY = "MASK_LOSS_KEY"
        self.RGB_RECONSTRUCTION_LOSS_KEY = "RGB_RECONSTRUCTION_LOSS_KEY"
        self.losses_dict_s[self.MASK_LOSS_KEY] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []
        self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[self.MASK_LOSS_KEY] = "Mask loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
        self.caption_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY] = "RGB Reconstruction loss per iteration"


    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        shadow_tensor = target_map["shadow"]

        input_ns = self.iid_op.remove_rgb_shadow(input_rgb_tensor, shadow_tensor, False)

        with amp.autocast():
            if (self.da_enabled == 1):
                input_ws = self.reshape_input(input_rgb_tensor)
            else:
                input_ws = input_rgb_tensor

            self.optimizerD_shading.zero_grad()

            # shadow map discriminator
            self.D_SM_discriminator.train()
            output = self.G_SM_predictor(input_ws)
            prediction = self.D_SM_discriminator(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.adversarial_loss(self.D_SM_discriminator(shadow_tensor), real_tensor) * self.adv_weight
            D_SM_fake_loss = self.adversarial_loss(self.D_SM_pool.query(self.D_SM_discriminator(output.detach())), fake_tensor) * self.adv_weight

            #RGB (no shadows) image discriminator
            self.D_rgb.train()
            output = self.G_rgb(input_ws)
            prediction = self.D_rgb(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_rgb_real_loss = self.adversarial_loss(self.D_rgb(input_ns), real_tensor) * self.adv_weight
            D_rgb_fake_loss = self.adversarial_loss(self.D_rgb_pool.query(self.D_rgb(output.detach())), fake_tensor) * self.adv_weight

            errD = D_SM_real_loss + D_SM_fake_loss + D_rgb_real_loss + D_rgb_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            #shadow map generator
            self.G_SM_predictor.train()
            rgb2sm = self.G_SM_predictor(input_ws)
            SM_likeness_loss = self.l1_loss(rgb2sm, shadow_tensor) * self.it_table.get_l1_weight(self.iteration)
            SM_lpip_loss = self.lpip_loss(rgb2sm, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration)
            SM_masking_loss = self.masking_loss(rgb2sm, shadow_tensor) * self.it_table.get_masking_weight(self.iteration)
            prediction = self.D_SM_discriminator(rgb2sm)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            #rgb refinement
            input_ns_like = self.iid_op.remove_rgb_shadow(input_ws, rgb2sm, False)
            input_ns_like = self.G_rgb(input_ns_like)
            RGB_recon_loss = self.l1_loss(input_ns_like, input_ns) * self.it_table.get_rgb_recon_weight()
            RGB_lpip_loss = self.lpip_loss(input_ns_like, input_ns) * self.it_table.get_rgb_lpips_weight()
            prediction = self.D_rgb(input_ns_like)
            real_tensor = torch.ones_like(prediction)
            RGB_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.it_table.get_rgb_adv_weight()

            errG = SM_likeness_loss + SM_lpip_loss + SM_masking_loss + SM_adv_loss + \
                RGB_recon_loss + RGB_lpip_loss + RGB_adv_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(SM_lpip_loss.item() + RGB_lpip_loss.item())
            self.losses_dict_s[self.MASK_LOSS_KEY].append(SM_masking_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(SM_adv_loss.item() + RGB_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item() + D_rgb_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item() + D_rgb_real_loss.item())
            self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY].append(RGB_recon_loss.item())

            _, rgb2shadow = self.test(input_map)
            self.stopper_method.register_metric(rgb2shadow, shadow_tensor, epoch)
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
            rgb2shadow = self.G_SM_predictor(input_ws)
            rgb2ns = self.iid_op.remove_rgb_shadow(input_ws, rgb2shadow, False)
            rgb2ns = self.G_rgb(rgb2ns)

        return rgb2ns, rgb2shadow

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        shadow_tensor = input_map["shadow"]
        rgb2ns, rgb2shadow = self.test(input_map)
        rgb2ns_equation = self.iid_op.remove_rgb_shadow(input_rgb_tensor, rgb2shadow, False)
        input_rgb_tensor_noshadow = self.iid_op.remove_rgb_shadow(input_rgb_tensor, shadow_tensor, False)

        shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)
        input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ns_equation = tensor_utils.normalize_to_01(rgb2ns_equation)
        rgb2shadow = tensor_utils.normalize_to_01(rgb2shadow)
        input_rgb_tensor_noshadow = tensor_utils.normalize_to_01(input_rgb_tensor_noshadow)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2shadow, str(label) + " Shadow-Like images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2ns_equation, str(label) + " RGB-NS (Equation) Images - " + self.NETWORK_VERSION + str(self.iteration))
        # self.visdom_reporter.plot_image(rgb2ns, str(label) + " RGB-NS (Generated) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(input_rgb_tensor_noshadow, str(label) + " RGB No Shadow Images - " + self.NETWORK_VERSION + str(self.iteration))


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
            self.G_rgb.load_state_dict(checkpoint[constants.GENERATOR_KEY + "WS"])
            self.D_rgb.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "WS"])
            self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "Z"])
            self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()
        netGWS_state_dict = self.G_rgb.state_dict()
        netDWS_state_dict = self.D_rgb.state_dict()

        optimizerGshading_state_dict = self.optimizerG_shading.state_dict()
        optimizerDshading_state_dict = self.optimizerD_shading.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()

        save_dict[constants.GENERATOR_KEY + "NS"] = netGNS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "NS"] = netDNS_state_dict
        save_dict[constants.GENERATOR_KEY + "WS"] = netGWS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "WS"] = netDWS_state_dict

        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"] = optimizerDshading_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "Z"] = schedulerGshading_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"] = schedulerDshading_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))