import kornia.metrics.psnr

from config import network_config
from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import shadow_matte_trainer, shadow_removal_trainer

class ShadowTester():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.sm_t = shadow_matte_trainer.ShadowMatteTrainer(self.gpu_device)
        self.sr_t = shadow_removal_trainer.ShadowTrainer(self.gpu_device)

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.mse_results = []
        self.rmse_results = []
        self.psnr_results = []
