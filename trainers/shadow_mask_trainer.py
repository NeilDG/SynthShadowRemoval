from config import iid_server_config
from trainers import abstract_iid_trainer, early_stopper
from model import unet_gan
import constants
import torch
import torch.cuda.amp as amp
import itertools
import torch.nn as nn
from transforms import iid_transforms
from utils import plot_utils

class ShadowMaskTrainer(abstract_iid_trainer.AbstractIIDTrainer):

    def __init__(self, gpu_device, opts):
        super().__init__(gpu_device, opts)

        self.initialize_train_config(opts)

    def initialize_train_config(self, opts):
        self.iteration = opts.iteration

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        general_config = sc_instance.get_general_configs()
        network_config = sc_instance.interpret_network_config_from_version()

        self.load_size = network_config["load_size_p"]
        self.batch_size = network_config["batch_size_p"]

        self.iid_op = iid_transforms.IIDTransform().to(self.gpu_device)
        self.fp16_scaler = amp.GradScaler()  # for automatic mixed precision

        min_epochs = general_config["train_shadow_mask"]["min_epochs"]
        self.stopper_method = early_stopper.EarlyStopper(min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, self.batch_size * 25, 99999.9)
        self.stop_result = False

        self.initialize_parsing_network(network_config["nc"])
        self.initialize_dict()

        self.NETWORK_VERSION = sc_instance.get_version_config("network_p_name", self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_parsing_network(self, input_nc):
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_P = network_creator.initialize_parsing_network(input_nc)
        self.optimizerP = torch.optim.Adam(itertools.chain(self.G_P.parameters()), lr=self.g_lr)
        self.schedulerP = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerP, patience=100000 / self.batch_size, threshold=0.00005)

    def initialize_dict(self):
        self.losses_dict_p = {}
        self.losses_dict_p[constants.LIKENESS_LOSS_KEY] = []

        self.caption_dict_p = {}
        self.caption_dict_p[constants.LIKENESS_LOSS_KEY] = "Classifier loss per iteration"

    def train(self, epoch, iteration, input_map, target_map):
        input_rgb_tensor = input_map["rgb"]
        mask_tensor = target_map["shadow_mask"]
        mask_tensor_test = target_map["mask_istd"]
        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            input = input_rgb_tensor

            mask_tensor_inv = 1 - mask_tensor
            output = torch.cat([mask_tensor, mask_tensor_inv], 1)
            self.G_P.train()
            self.optimizerP.zero_grad()

            # print("Shapes: ", np.shape(self.G_P(input)), np.shape(output))
            mask_loss = self.bce_loss(self.G_P(input), output)
            self.fp16_scaler.scale(mask_loss).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerP)
                self.schedulerP.step(mask_loss)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                self.losses_dict_p[constants.LIKENESS_LOSS_KEY].append(mask_loss.item())

                self.stopper_method.register_metric(self.test_istd(input_map), mask_tensor_test, epoch)
                self.stop_result = self.stopper_method.test(epoch)

        if(self.stopper_method.has_reset()):
            self.save_states(epoch, iteration, False)

    def test_istd(self, input_map):
        # print("Testing on ISTD dataset.")
        input_map_new = {"rgb" : input_map["rgb_ws_istd"]}
        return self.test(input_map_new)

    def is_stop_condition_met(self):
        return self.stop_result

    def test(self, input_map):
        with torch.no_grad():
            input_rgb_tensor = input_map["rgb"]
            input = input_rgb_tensor

            rgb2mask = self.G_P(input)
            rgb2mask = torch.round(rgb2mask)[:, 0, :, :]
            rgb2mask = torch.unsqueeze(rgb2mask, 1)
            return rgb2mask

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_p", iteration, self.losses_dict_p, self.caption_dict_p, self.NETWORK_CHECKPATH)

    def visdom_visualize(self, input_map, label="Train"):
        with torch.no_grad():
            input_rgb_tensor = input_map["rgb"]
            rgb2mask = self.test(input_map)

            self.visdom_reporter.plot_image(input_rgb_tensor, str(label) + " Input RGB Images - " + self.NETWORK_VERSION + str(self.iteration))
            self.visdom_reporter.plot_image(rgb2mask, str(label) + " Shadow-Mask-Like - " + self.NETWORK_VERSION + str(self.iteration))

            if("shadow_mask" in input_map):
                mask_tensor = input_map["shadow_mask"]
                self.visdom_reporter.plot_image(mask_tensor, str(label) + " Shadow Masks - " + self.NETWORK_VERSION + str(self.iteration))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
            print("Loaded network: ", self.NETWORK_CHECKPATH)
        except:
            #check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("Loaded network: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            iid_server_config.IIDServerConfig.getInstance().store_epoch_from_checkpt("train_shadow_mask", checkpoint["epoch"])
            self.stopper_method.update_last_metric(checkpoint[constants.LAST_METRIC_KEY])
            self.G_P.load_state_dict(checkpoint[constants.GENERATOR_KEY + "P"])
            self.optimizerP.load_state_dict(checkpoint[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "P"])
            self.schedulerP.load_state_dict(checkpoint[constants.GENERATOR_KEY + "scheduler" + "P"])


    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, constants.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}

        netGP_state_dict = self.G_P.state_dict()
        optimizerP_state_dict = self.optimizerP.state_dict()
        schedulerP_state_dict = self.schedulerP.state_dict()
        save_dict[constants.GENERATOR_KEY + "P"] = netGP_state_dict
        save_dict[constants.GENERATOR_KEY + constants.OPTIMIZER_KEY + "P"] = optimizerP_state_dict
        save_dict[constants.GENERATOR_KEY + "scheduler" + "P"] = schedulerP_state_dict

        if(is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))


        
        
