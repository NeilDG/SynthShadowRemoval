from abc import abstractmethod

import torch

import constants
from model import embedding_network


class AbstractIIDTrainer():
    def __init__(self, gpu_device, opts):
        self.gpu_device = gpu_device
        self.g_lr = opts.g_lr
        self.d_lr = opts.d_lr

    def assign_embedder_decoder(self, embedder, decoder):
        self.embedder = embedder
        self.decoder = decoder

    def reshape_input(self, input_tensor):
        rgb_embedding, w1, w2, w3 = self.embedder.get_embedding(input_tensor)
        rgb_feature_rep = self.decoder.get_decoding(input_tensor, rgb_embedding, w1, w2, w3)

        return torch.cat([input_tensor, rgb_feature_rep], 1)

    def get_feature_rep(self, input_tensor):
        rgb_embedding, w1, w2, w3 = self.embedder.get_embedding(input_tensor)
        rgb_feature_rep = self.decoder.get_decoding(input_tensor, rgb_embedding, w1, w2, w3)

        return rgb_feature_rep

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
