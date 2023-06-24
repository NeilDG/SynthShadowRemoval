# Class for simple detection for best metric. If a better metric is found, it returns true.
import kornia
import numpy as np
import torch
from torch import nn

from config.network_config import ConfigHolder
from trainers.early_stopper import EarlyStopperMethod

class BestTracker():
    def __init__(self, current_epoch, early_stopper_method, last_metric = 10000.0):
        self.best_metric = last_metric
        if (early_stopper_method is EarlyStopperMethod.L1_TYPE):
            self.loss_op = nn.L1Loss()
        elif (early_stopper_method is EarlyStopperMethod.SSIM_TYPE):
            self.loss_op = kornia.losses.SSIMLoss(5)

        self.start_epoch = current_epoch
        self.max_epoch_tolerance = ConfigHolder.getInstance().get_network_attribute("epoch_tolerance", 5)
        self.best_metric_replaced = False

    def test(self, input, target):
        loss_result = self.loss_op(input, target).item()

        if(loss_result < self.best_metric):
            self.best_metric = loss_result
            self.best_metric_replaced = True
            return True
        else:
            return False

    def has_plateau(self, epoch):
        if(self.best_metric_replaced == True):
            self.start_epoch = epoch
            self.best_metric_replaced = False

        max_epoch = self.start_epoch + self.max_epoch_tolerance
        if(epoch >= self.start_epoch + self.max_epoch_tolerance):
            print("Training has plateaued! Should stop!!! Last epoch: ", epoch, "Max epoch tolerance: ", max_epoch)
            return True
        else:
            print("Training hasn't plateaued. Continuing. Last epoch: ", epoch, "Max epoch tolerance: ", max_epoch)
            return False


    def reset(self):
        self.best_metric = 100000.0

    def get_best_metric(self):
        return np.round(self.best_metric, 4)

    def load_best_state(self, network_file_name):
        try:
            checkpoint = torch.load(network_file_name)
        except:
            checkpoint = None
            print("No existing checkpoint file found. Not updating best state. ", network_file_name)

        if(checkpoint != None and "best_metric" in checkpoint):
            self.best_metric = checkpoint["best_metric"]
            print("Updated best metric from file: ", self.best_metric)