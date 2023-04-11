import global_config
from loaders import dataset_loader
from model import embedding_network
from model import ffa_gan as ffa
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from trainers import early_stopper
from trainers.shadow_matte_trainer import ShadowMatteTrainer
# from trainers.shadow_trainer import ShadowTrainer
from trainers.shadow_removal_trainer import ShadowTrainer
import torch

class TrainerFactory():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.trainer_list = {}

    # def initialize_all_trainers(self):
    #     # self.trainer_list["train_shadow_mask"] = ShadowMaskTrainer(self.gpu_device, opts)
    #     self.trainer_list["train_shadow_matte"] = ShadowMatteTrainer(self.gpu_device)
    #     self.trainer_list["train_shadow"] = ShadowTrainer(self.gpu_device)
    #     # self.trainer_list["train_shadow_refine"] = ShadowRefinementTrainer(self.gpu_device, opts)

    # def get_all_trainers(self):
    #     self.initialize_all_trainers()
    #     return self.trainer_list["train_shadow_matte"], self.trainer_list["train_shadow"]

    def get_shadow_trainer(self):
        if ("train_shadow" in self.trainer_list):
            return self.trainer_list["train_shadow"]
        else:
            self.trainer_list["train_shadow"] = ShadowTrainer(self.gpu_device)
            return self.trainer_list["train_shadow"]

    def get_shadow_matte_trainer(self):
        if ("train_shadow_matte" in self.trainer_list):
            return self.trainer_list["train_shadow_matte"]
        else:
            self.trainer_list["train_shadow_matte"] = ShadowMatteTrainer(self.gpu_device)
            return self.trainer_list["train_shadow_matte"]
    #
    # def train(self, mode, epoch, iteration, input_map, target_map):
    #     self.trainer_list[mode].train(epoch, iteration, input_map, target_map)
    #
    # def test(self, mode, input_map):
    #     self.trainer_list[mode].test(input_map)
    #
    # def is_stop_condition_met(self, mode):
    #     return self.trainer_list[mode].is_stop_condition_met()
    #
    # def visdom_plot(self, mode, iteration):
    #     if (self.trainer_list[mode] != None):
    #         self.trainer_list[mode].visdom_plot(iteration)
    #
    # def visdom_visualize(self, mode, input_map, label = "Train"):
    #     if(self.trainer_list[mode] != None):
    #         self.trainer_list[mode].visdom_visualize(input_map, label)
    #
    # def visdom_infer(self, mode, input_map):
    #     if(self.trainer_list[mode] != None):
    #         self.trainer_list[mode].visdom_infer(input_map)
    #
    # def save(self, mode, epoch, iteration, is_temp:bool):
    #     if(self.trainer_list[mode] != None):
    #         self.trainer_list[mode].save_states(epoch, iteration, is_temp)

