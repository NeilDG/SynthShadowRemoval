from abc import abstractmethod

import torch

import constants
from model import embedding_network


class AbstractIIDTrainer():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

    def initialize_da_network(self, da_version_name):
        self.embedder = embedding_network.EmbeddingNetworkFFA(blocks=6).to(self.gpu_device)
        checkpoint = torch.load("checkpoint/" + da_version_name + ".pt", map_location=self.gpu_device)
        self.embedder.load_state_dict(checkpoint[constants.GENERATOR_KEY + "A"])
        print("Loaded embedding network: ", da_version_name)

        self.decoder_fixed = embedding_network.DecodingNetworkFFA().to(self.gpu_device)
        print("Loaded fixed decoder network")

    @abstractmethod
    def initialize_train_config(self, opts):
        pass

    @abstractmethod
    def initialize_dict(self):
        # what to store in visdom?
        pass

    @abstractmethod
    #follows a hashmap style lookup
    def train(self, epoch, iteration, input_map, target_map):
        pass

    @abstractmethod
    def is_stop_condition_met(self):
        pass

    @abstractmethod
    def test(self, input_map):
        pass

    @abstractmethod
    def visdom_plot(self, iteration):
        pass

    @abstractmethod
    def visdom_visualize(self, input_map, label="Train"):
        pass

    @abstractmethod
    def visdom_infer(self, input_map):
        pass

    @abstractmethod
    def load_saved_state(self):
        pass

    @abstractmethod
    def save_states(self, epoch, iteration, is_temp:bool):
        pass
