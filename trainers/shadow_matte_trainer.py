import torchvision.transforms.functional

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
from transforms import iid_transforms, shadow_map_transforms
from utils import plot_utils
from utils import tensor_utils
from custom_losses import ssim_loss, iid_losses
import lpips
from model.modules import shadow_matte_pool

class ShadowMatteTrainer(abstract_iid_trainer.AbstractIIDTrainer):
    def __init__(self, gpu_device, opts):
        super().__init__(gpu_device, opts)
        self.initialize_train_config(opts)

    def initialize_train_config(self, opts):
        self.iteration = opts.shadow_matte_iteration
        self.it_table = shadow_iteration_table.ShadowIterationTable()
        self.use_bce = self.it_table.is_bce_enabled(self.iteration)
        self.adv_weight = self.it_table.get_adv_weight()

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.gpu_device)
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.D_SM_pool = image_pool.ImagePool(50)

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        network_config = sc_instance.interpret_shadow_matte_params_from_version()
        general_config = sc_instance.get_general_configs()

        self.load_size = network_config["load_size_m"]
        self.batch_size = network_config["batch_size_m"]
        self.patch_size = general_config["train_shadow_matte"]["patch_size"]
        self.use_grayscale = network_config["use_grayscale"]
        self.use_istd_pool = network_config["use_istd_pool"]

        if(self.use_istd_pool):
            shadow_matte_pool.ShadowMattePool.initialize()
            self.ISTD_SM_pool = shadow_matte_pool.ShadowMattePool().getInstance()

        self.stopper_method = early_stopper.EarlyStopper(general_config["train_shadow_matte"]["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, 3000, 99999.9)
        self.stop_result = False

        self.initialize_dict()
        self.initialize_shadow_network(network_config["net_config"], network_config["num_blocks"], network_config["nc"])

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_SM_predictor.parameters()), lr=self.g_lr, weight_decay=network_config["weight_decay"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_SM_discriminator.parameters()), lr=self.d_lr, weight_decay=network_config["weight_decay"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = sc_instance.get_version_config("shadow_matte_network_version", "network_m_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_shadow_network(self, net_config, num_blocks, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_SM_predictor, self.D_SM_discriminator = network_creator.initialize_shadow_matte_network(net_config, num_blocks, input_nc)  # shadow map (Shadow - Shadow-Free)

    def adversarial_loss(self, pred, target):
        if (self.use_bce == 0):
            return self.mse_loss(pred, target)
        else:
            return self.bce_loss(pred, target)

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

        return self.l1_loss(pred_mask, target_mask)


    def istd_sm_loss(self, pred):
        if(self.use_istd_pool):
            istd_sm_samples = self.ISTD_SM_pool.query_samples(np.shape(pred)[0])
            # print("Shape of ISTD SM samples:" , np.shape(istd_sm_samples), np.shape(pred))
            return self.l1_loss(pred, istd_sm_samples)
        else:
            return self.l1_loss(pred, pred) #workaround to return 0.0

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
        self.MASK_LOSS_KEY = "MASK_LOSS_KEY"
        self.losses_dict_s[self.MASK_LOSS_KEY] = []
        self.ISTD_SM_LOSS_KEY = "ISTD_SM_LOSS_KEY"
        self.losses_dict_s[self.ISTD_SM_LOSS_KEY] = []

        self.caption_dict_s = {}
        self.caption_dict_s[constants.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict_s[constants.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict_s[constants.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict_s[constants.LPIP_LOSS_KEY] = "LPIPS loss per iteration"
        self.caption_dict_s[constants.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict_s[constants.D_A_FAKE_LOSS_KEY] = "D fake loss per iteration"
        self.caption_dict_s[constants.D_A_REAL_LOSS_KEY] = "D real loss per iteration"
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
        if(self.use_grayscale):
            input_ws = input_map["rgb_ws_gray"]
        else:
            input_ws = input_map["rgb"]
        target_tensor = target_map["shadow_matte"]
        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            if(self.use_istd_pool):
                shadow_matte_pool.ShadowMattePool.getInstance().set_samples(input_map["matte_train_istd"])

            # shadow map discriminator
            self.optimizerD.zero_grad()
            self.D_SM_discriminator.train()
            output = self.G_SM_predictor(input_ws)
            prediction = self.D_SM_discriminator(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.adversarial_loss(self.D_SM_discriminator(target_tensor), real_tensor) * self.adv_weight
            D_SM_fake_loss = self.adversarial_loss(self.D_SM_pool.query(self.D_SM_discriminator(output.detach())), fake_tensor) * self.adv_weight

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)

            #shadow map generator
            self.optimizerG.zero_grad()
            self.G_SM_predictor.train()
            rgb2sm = self.G_SM_predictor(input_ws)
            SM_likeness_loss = self.l1_loss(rgb2sm, target_tensor) * self.it_table.get_l1_weight(self.iteration)
            SM_lpip_loss = self.lpip_loss(rgb2sm, target_tensor) * self.it_table.get_lpip_weight(self.iteration)
            SM_masking_loss = self.masking_loss(rgb2sm, target_tensor) * self.it_table.get_masking_weight(self.iteration)
            prediction = self.D_SM_discriminator(rgb2sm)
            real_tensor = torch.ones_like(prediction)
            SM_adv_loss = self.adversarial_loss(prediction, real_tensor) * self.adv_weight
            SM_istd_loss = self.istd_sm_loss(rgb2sm)

            errG = SM_likeness_loss + SM_lpip_loss + SM_masking_loss + SM_adv_loss + SM_istd_loss

            self.fp16_scaler.scale(errG).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerG)
                self.schedulerG.step(errG)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                self.losses_dict_s[constants.G_LOSS_KEY].append(errG.item())
                self.losses_dict_s[constants.D_OVERALL_LOSS_KEY].append(errD.item())
                self.losses_dict_s[constants.LIKENESS_LOSS_KEY].append(SM_likeness_loss.item())
                self.losses_dict_s[constants.LPIP_LOSS_KEY].append(SM_lpip_loss.item())
                self.losses_dict_s[constants.G_ADV_LOSS_KEY].append(SM_adv_loss.item())
                self.losses_dict_s[constants.D_A_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
                self.losses_dict_s[constants.D_A_REAL_LOSS_KEY].append(D_SM_real_loss.item())
                self.losses_dict_s[self.MASK_LOSS_KEY].append(SM_masking_loss.item())
                self.losses_dict_s[self.ISTD_SM_LOSS_KEY].append(SM_istd_loss.item())

                #perform validation test and early stopping
                rgb2sm_istd = self.test_istd(input_map)
                istd_sm_test = input_map["matte_istd"]
                self.stopper_method.register_metric(rgb2sm_istd, istd_sm_test, epoch)
                self.stop_result = self.stopper_method.test(epoch)

                if (self.stopper_method.has_reset()):
                    self.save_states(epoch, iteration, False)

                #plot train-test loss
                rgb2sm = self.test(input_map)
                rgb2sm = tensor_utils.normalize_to_01(rgb2sm)
                target_tensor = tensor_utils.normalize_to_01(target_tensor)
                rgb2sm_istd = tensor_utils.normalize_to_01(rgb2sm_istd)
                istd_sm_test = tensor_utils.normalize_to_01(istd_sm_test)
                self.losses_dict_t[self.TRAIN_LOSS_KEY].append(self.l1_loss(rgb2sm, target_tensor).item())
                self.losses_dict_t[self.TEST_LOSS_KEY].append(self.l1_loss(rgb2sm_istd, istd_sm_test).item())

    def is_stop_condition_met(self):
        return self.stop_result

    def test_istd(self, input_map):
        # print("Testing on ISTD dataset.")

        input_map_new = {"rgb": input_map["rgb_ws_istd"],
                         "rgb_ws_gray" : input_map["gray_istd"],
                         "shadow_matte": input_map["matte_istd"]}

        return self.test(input_map_new)

    def test(self, input_map):
        with torch.no_grad():
            self.G_SM_predictor.eval()
            if (self.use_grayscale):
                input_ws = input_map["rgb_ws_gray"]
            else:
                input_ws = input_map["rgb"]

            rgb2sm = self.G_SM_predictor(input_ws)
            # rgb2sm = 1.0 - rgb2sm

        return rgb2sm

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict_s, self.caption_dict_s, self.NETWORK_CHECKPATH)
        self.visdom_reporter.plot_train_test_loss("train_test_loss", iteration, self.losses_dict_t, self.caption_dict_t, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        if (self.use_grayscale):
            input_ws = input_map["rgb_ws_gray"]
        else:
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
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_shadow_matte", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_SM_predictor.load_state_dict(checkpoint[constants.GENERATOR_KEY + "M"])
            self.D_SM_discriminator.load_state_dict(checkpoint[constants.DISCRIMINATOR_KEY + "M"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGNS_state_dict = self.G_SM_predictor.state_dict()
        netDNS_state_dict = self.D_SM_discriminator.state_dict()

        save_dict[constants.GENERATOR_KEY + "M"] = netGNS_state_dict
        save_dict[constants.DISCRIMINATOR_KEY + "M"] = netDNS_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))












