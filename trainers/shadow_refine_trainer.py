import torchvision.transforms as transforms
from torchvision.transforms import functional as transform_functional
from config import iid_server_config
from trainers import abstract_iid_trainer, early_stopper, trainer_factory
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

class ShadowRefineTrainer(abstract_iid_trainer.AbstractIIDTrainer):
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

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision
        self.iid_op = iid_transforms.IIDTransform()
        self.norm_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_network_config_from_version()

        self.batch_size = network_config["batch_size_zr"]
        self.stopper_method = early_stopper.EarlyStopper(general_config["train_refine_shadow"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shadow_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.optimizerG_refine = torch.optim.Adam(itertools.chain(self.G_refiner.parameters()), lr=self.g_lr)
        self.optimizerD_refine = torch.optim.Adam(itertools.chain(self.D_SM_discriminator.parameters()), lr=self.d_lr)
        self.schedulerG_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG_refine, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD_shading = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_refine, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("network_zr_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_refiner, self.D_SM_discriminator = network_creator.initialize_rgb_network(net_config, num_blocks, input_nc)

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

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict_s = {}
        self.losses_dict_s[constants.G_LOSS_KEY] = []
        self.losses_dict_s[constants.D_OVERALL_LOSS_KEY] = []
        self.losses_dict_s[constants.LIKENESS_LOSS_KEY] = []
        self.losses_dict_s[constants.LPIP_LOSS_KEY] = []
        self.losses_dict_s[constants.G_ADV_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY] = []
        self.losses_dict_s[constants.D_A_REAL_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Refine G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        # input_ws = tensor_utils.normalize_to_01(input_map["rgb"])
        # rgb_ws_relit = tensor_utils.normalize_to_01(input_map["rgb_relit"])
        # shadow_matte = tensor_utils.normalize_to_01(input_map["shadow_matte"])
        # input_refined = self.iid_op.remove_shadow(input_ws, rgb_ws_relit, shadow_matte, -1.0, 1.0)
        # input_refined = self.norm_op(input_refined)
        input_refined = input_map["rgb"]
        output_ns = target_map["rgb_ns"]

        with amp.autocast():
            #discriminator
            self.optimizerD_refine.zero_grad()
            self.D_SM_discriminator.train()
            output = self.G_refiner(input_refined)
            prediction = self.D_SM_discriminator(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.adversarial_loss(self.D_SM_discriminator(output_ns), real_tensor) * self.adv_weight
            D_SM_fake_loss = self.adversarial_loss(self.D_SM_pool.query(self.D_SM_discriminator(output.detach())), fake_tensor) * self.adv_weight

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            self.fp16_scaler.step(self.optimizerD_refine)
            self.schedulerD_shading.step(errD)

            # refiner
            self.optimizerG_refine.zero_grad()
            self.G_refiner.train()
            ns_like = self.G_refiner(input_refined)
            SM_likeness_loss = self.l1_loss(ns_like, output_ns) * self.it_table.get_l1_weight(self.iteration)
            SM_lpip_loss = self.lpip_loss(ns_like, output_ns) * self.it_table.get_lpip_weight(self.iteration)
            prediction = self.D_SM_discriminator(ns_like)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight

            errG = SM_likeness_loss + SM_lpip_loss + SM_adv_loss

            self.fp16_scaler.scale(errG).backward()
            self.fp16_scaler.step(self.optimizerG_refine)
            self.schedulerG_shading.step(errG)
            self.fp16_scaler.update()

            # what to put to losses dict for visdom reporting?
            self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
            self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
            self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
            self.losses_dict_s[constants.LPIP_LOSS_KEY].append(SM_lpip_loss.item())
            self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(SM_adv_loss.item())
            self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
            self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item())\

    def test(self, input_map):
        with torch.no_grad():
            input_ws = input_map["rgb"]
            #use shadow trainer model
            tf = trainer_factory.TrainerFactory.getInstance()
            shadow_t = tf.get_shadow_trainer()
            _, shadow_matte, rgb_ws_relit = shadow_t.test(input_map)

            input_ws = tensor_utils.normalize_to_01(input_ws)
            rgb_ws_relit = tensor_utils.normalize_to_01(rgb_ws_relit)
            shadow_matte = tensor_utils.normalize_to_01(shadow_matte)

            input_refined = self.iid_op.remove_shadow(input_ws, rgb_ws_relit, shadow_matte)
            input_refined = transform_functional.normalize(input_refined, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ns_like = self.G_refiner(input_refined)

            return input_refined, ns_like

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        # input_ws = input_map["rgb"]
        input_ns = input_map["rgb_ns"]
        input_ws, ns_like = self.test(input_map)

        input_ws = tensor_utils.normalize_to_01(input_ws)
        input_ns = tensor_utils.normalize_to_01(input_ns)
        ns_like = tensor_utils.normalize_to_01(ns_like)

        self.visdom_reporter.plot_image(input_ws, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(ns_like, str(label) + " RGB-NS (Refined) Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(input_ns, str(label) + " RGB NS Images - " + self.NETWORK_VERSION + str(self.iteration))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
            print("Loaded shadow refine network: ", self.NETWORK_CHECKPATH)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("Loaded shadow refine network: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new shadow refine network: ", self.NETWORK_CHECKPATH)

        if(checkpoint != None):
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_refine_shadow", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_refiner.load_state_dict(checkpoint[constants.GENERATOR_KEY + "NS"])
            self.D_SM_discriminator.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "NS"])

            self.optimizerG_refine.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.optimizerD_refine.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + constants.OPTIMIZER_KEY + "Z"])
            self.schedulerG_shading.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "Z"])
            self.schedulerD_shading.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "scheduler" + "Z"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_refiner.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()

        optimizerGshading_state_dict = self.optimizerG_refine.state_dict()
        optimizerDshading_state_dict = self.optimizerD_refine.state_dict()
        schedulerGshading_state_dict = self.schedulerG_shading.state_dict()
        schedulerDshading_state_dict = self.schedulerD_shading.state_dict()

        save_dict[constants.GENERATOR_KEY + "NS"] = netGNS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "NS"] = netDNS_state_dict

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