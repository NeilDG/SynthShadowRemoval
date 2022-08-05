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

        self.D_NS_pool = image_pool.ImagePool(50)
        self.D_WS_pool = image_pool.ImagePool(50)

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

        self.optimizerG_shading = torch.optim.Adam(itertools.chain(self.G_NS.parameters(), self.G_WS.parameters()), lr=self.g_lr)
        self.optimizerD_shading = torch.optim.Adam(itertools.chain(self.D_NS.parameters(), self.D_WS.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_shading, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_shading, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_z_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_NS, self.D_NS = network_creator.initialize_shadow_network(net_config, num_blocks, input_nc)
        self.G_WS, self.D_WS = network_creator.initialize_shadow_network(net_config, num_blocks, input_nc)

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
        self.losses_dict_s[constants.CYCLE_LOSS_KEY] = []
        self.SHADOW_SIM_KEY = "SHADOPW_SIM_KEY"
        self.losses_dict_s[self.SHADOW_SIM_KEY] = []

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
        self.caption_dict_s[constants.CYCLE_LOSS_KEY] = "Cycle loss per iteration"
        self.caption_dict_s[self.SHADOW_SIM_KEY] = "Shadow sim per iteration"


    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        shadow_tensor = target_map["shadow"]
        input_rgb_tensor_noshadow = self.iid_op.remove_rgb_shadow(input_rgb_tensor, shadow_tensor, False)

        with amp.autocast():
            if (self.da_enabled == 1):
                input_ws = self.reshape_input(input_rgb_tensor)
                input_ns = self.reshape_input(input_rgb_tensor_noshadow)
            else:
                input_ws = input_rgb_tensor
                input_ns = input_rgb_tensor_noshadow

            self.optimizerD_shading.zero_grad()

            # shadow discriminator
            self.D_NS.train()
            output = self.G_NS(input_ws)
            prediction = self.D_NS(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_NS_real_loss = self.adversarial_loss(self.D_NS(torch.cat([input_ns, shadow_tensor], 1)), real_tensor) * self.adv_weight
            D_NS_fake_loss = self.adversarial_loss(self.D_NS_pool.query(self.D_NS(output.detach())), fake_tensor) * self.adv_weight

            self.D_WS.train()
            output = self.G_WS(input_ns)
            prediction = self.D_WS(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_WS_real_loss = self.adversarial_loss(self.D_WS(torch.cat([input_ws, shadow_tensor], 1)), real_tensor) * self.adv_weight
            D_WS_fake_loss = self.adversarial_loss(self.D_WS_pool.query(self.D_WS(output.detach())), fake_tensor) * self.adv_weight

            errD = D_NS_real_loss + D_NS_fake_loss + D_WS_real_loss + D_WS_fake_loss
            self.fp16_scaler.scale(errD).backward()
            if (self.fp16_scaler.scale(errD).item() > 0.0):
                self.fp16_scaler.step(self.optimizerD_shading)
                self.schedulerD_shading.step(errD)

            self.optimizerG_shading.zero_grad()

            # shadow generator
            self.G_NS.train()
            self.G_WS.train()

            output = self.G_NS(input_ws)
            rgb2ns, rgb2shadow = torch.split(output, [3, 1], 1)
            NS_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            NS_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            NS_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            NS_gradient_loss = self.gradient_loss_term(rgb2shadow, shadow_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            prediction = self.D_NS(output)
            real_tensor = torch.ones_like(prediction)
            NS_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
            rgb_ns_equation = self.iid_op.remove_rgb_shadow(input_rgb_tensor, rgb2shadow, False)
            NS_rgb_loss = (self.l1_loss(rgb_ns_equation, input_rgb_tensor_noshadow) + self.l1_loss(rgb2ns, input_rgb_tensor_noshadow)) * self.rgb_l1_weight

            output = self.G_WS(input_ns)
            rgb2ws, rgb2shadow = torch.split(output, [3, 1], 1)
            WS_likeness_loss = self.l1_loss(rgb2shadow, shadow_tensor) * self.it_table.get_l1_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            WS_lpip_loss = self.lpip_loss(rgb2shadow, shadow_tensor) * self.it_table.get_lpip_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            WS_ssim_loss = self.ssim_loss(rgb2shadow, shadow_tensor) * self.it_table.get_ssim_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            WS_gradient_loss = self.gradient_loss_term(rgb2shadow, shadow_tensor) * self.it_table.get_gradient_weight(self.iteration, IterationTable.NetworkType.SHADOW)
            prediction = self.D_WS(output)
            real_tensor = torch.ones_like(prediction)
            WS_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
            rgb_ws_equation = self.iid_op.add_rgb_shadow(input_rgb_tensor_noshadow, rgb2shadow, False)
            WS_rgb_loss = (self.l1_loss(rgb_ws_equation, input_rgb_tensor) + self.l1_loss(rgb2ws, input_rgb_tensor)) * self.rgb_l1_weight

            #cycleloss
            rgb2ws, rgb2shadow = torch.split(self.G_WS(input_ns), [3, 1], 1)
            ns_output = torch.cat([input_ns, shadow_tensor], 1)
            NS_cycle_loss = self.l1_loss(self.G_NS(rgb2ws), ns_output) * 10.0

            rgb2ns, rgb2shadow = torch.split(self.G_NS(input_ws), [3, 1], 1)
            ws_output = torch.cat([input_ws, shadow_tensor], 1)
            WS_cycle_loss = self.l1_loss(self.G_WS(rgb2ns), ws_output) * 10.0

            #reduce shadow difference
            Z_sim_loss = self.l1_loss(rgb2ws, rgb2ns) * 20.0

            errG = NS_likeness_loss + NS_lpip_loss + NS_ssim_loss + NS_gradient_loss + NS_adv_loss + NS_rgb_loss + \
                   WS_likeness_loss + WS_lpip_loss + WS_ssim_loss + WS_gradient_loss + WS_adv_loss + WS_rgb_loss + \
                   NS_cycle_loss + WS_cycle_loss + Z_sim_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_shading)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(NS_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(NS_lpip_loss.item())
            self.losses_dict_s[constants.SSIM_LOSS_KEY].append(NS_ssim_loss.item())
            self.losses_dict_s[self.GRADIENT_LOSS_KEY].append(NS_gradient_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(NS_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_NS_fake_loss.item() + D_WS_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_NS_real_loss.item() + D_WS_real_loss.item())
            self.losses_dict_s[self.RGB_RECONSTRUCTION_LOSS_KEY].append(NS_rgb_loss.item() + WS_rgb_loss.item())
            self.losses_dict_s[constants.CYCLE_LOSS_KEY].append(NS_cycle_loss.item() + WS_cycle_loss.item())
            self.losses_dict_s[self.SHADOW_SIM_KEY].append(Z_sim_loss.item())

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

            output = self.G_NS(input_ws)
            rgb2ns, rgb2shadow = torch.split(output, [3, 1], 1)

        return rgb2ns, rgb2shadow

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        input_rgb_tensor = input_map["rgb"]
        shadow_tensor = input_map["shadow"]
        input_rgb_tensor_noshadow = self.iid_op.remove_rgb_shadow(input_rgb_tensor, shadow_tensor, False)

        if (self.da_enabled == 1):
            input_ns = self.reshape_input(input_rgb_tensor_noshadow)
        else:
            input_ns = input_rgb_tensor_noshadow

        embedding_rep = self.get_feature_rep(input_rgb_tensor)
        rgb2ns, rgb2shadow_ns = self.test(input_map)
        output = self.G_WS(input_ns)
        rgb2ws, rgb2shadow_ws = torch.split(output, [3, 1], 1)

        shadow_tensor = tensor_utils.normalize_to_01(shadow_tensor)
        input_rgb_tensor = tensor_utils.normalize_to_01(input_rgb_tensor)
        input_rgb_tensor_noshadow = tensor_utils.normalize_to_01(input_rgb_tensor_noshadow)
        rgb2shadow_ns = tensor_utils.normalize_to_01(rgb2shadow_ns)
        rgb2shadow_ws = tensor_utils.normalize_to_01(rgb2shadow_ws)
        rgb2ns = tensor_utils.normalize_to_01(rgb2ns)
        rgb2ws = tensor_utils.normalize_to_01(rgb2ws)

        rgb2ws_equation = self.iid_op.add_rgb_shadow(input_rgb_tensor_noshadow, rgb2shadow_ws, False)
        rgb2ns_equation = self.iid_op.remove_rgb_shadow(input_rgb_tensor, rgb2shadow_ns, False)

        self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2ws_equation, str(label) + " RGB-WS (Equation) Images " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2ws, str(label) + " RGB-WS (Generated) Images " + self.NETWORK_VERSION + str(self.iteration))
        # self.visdom_reporter.plot_image(embedding_rep, str(label) + " Embedding Maps - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2shadow_ns, str(label) + " RGB2Shadow G_NS(Z) images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2shadow_ws, str(label) + " RGB2Shadow G_WS(Z) images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(shadow_tensor, str(label) + " Shadow images - " + self.NETWORK_VERSION + str(self.iteration))

        self.visdom_reporter.plot_image(rgb2ns_equation, str(label) + " RGB-NS (Equation) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(rgb2ns, str(label) + " RGB-NS (Generated) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(input_rgb_tensor_noshadow, str(label) + " RGB No Shadow Images - " + self.NETWORK_VERSION + str(self.iteration))


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
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_shadow", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_NS.load_state_dict(checkpoint[constants.GENERATOR_KEY + "NS"])
            self.D_NS.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "NS"])
            self.G_WS.load_state_dict(checkpoint[constants.GENERATOR_KEY + "WS"])
            self.D_WS.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "WS"])
            self.optimizerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.optimizerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "Z"])
            self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_NS.state_dict()
        netDNS_state_dict = self.D_NS.state_dict()
        netGWS_state_dict = self.G_WS.state_dict()
        netDWS_state_dict = self.D_WS.state_dict()

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