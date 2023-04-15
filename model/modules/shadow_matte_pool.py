import glob
import itertools

import kornia
import numpy as np
import torch
import torchvision.utils
from torch.utils import data
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm

import global_config
from loaders import image_datasets, shadow_datasets
import os
from loaders import dataset_loader

class ShadowMattePool():
    _sharedInstance = None

    @staticmethod
    def initialize():
        if (ShadowMattePool._sharedInstance == None):
            ShadowMattePool._sharedInstance = ShadowMattePool()

    @staticmethod
    def getInstance():
        return ShadowMattePool._sharedInstance

    def __init__(self):
        print("Init shadow matte pool")
        sc_instance = iid_server_config.IIDServerConfig.getInstance()
        network_config = sc_instance.interpret_shadow_matte_params_from_version()
        load_size = network_config["load_size_m"]
        self.pool_size = load_size
        self.sm_pool = None

    def set_samples(self, matte_istd):
        if (self.sm_pool == None):
            self.sm_pool = matte_istd[0]

        for i in range(0, np.shape(matte_istd)[0]):
            if (len(self.sm_pool) < self.pool_size):
                self.sm_pool = torch.cat([self.sm_pool, matte_istd[i]], 0)
            else:
                #replace random SM in pool
                # new_index = np.random.randint(0, self.pool_size)
                # self.sm_pool[new_index] = matte_istd[i]

                # replace index sequentially
                self.sm_pool[i] = matte_istd[i]

    def query_samples(self, query_size):
        return torch.unsqueeze(self.sm_pool[0: query_size], 1)



