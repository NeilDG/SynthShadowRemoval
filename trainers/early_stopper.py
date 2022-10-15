# -*- coding: utf-8 -*-
# Class for early stopping mechanism
from enum import Enum
import torch.nn as nn
import kornia
import torch.cuda.amp as amp
import torch

class EarlyStopperMethod(Enum):
    L1_TYPE = 0,
    SSIM_TYPE = 1,
    PSNR_TYPE = 2,

class EarlyStopper():

    class TestMetric():
        def __init__(self, input, target):
            self.input = input
            self.target = target

        def get_input(self):
            return self.input

        def get_target(self):
            return self.target

    def __init__(self, min_epochs, early_stopper_method, early_stop_tolerance, last_metric = 10000.0):
        self.min_epochs = min_epochs
        self.early_stop_tolerance = early_stop_tolerance
        self.stop_counter = 0
        self.last_metric = last_metric
        self.stop_condition_met = False
        self.network = None

        if(early_stopper_method is EarlyStopperMethod.L1_TYPE):
            self.loss_op = nn.L1Loss()
        elif(early_stopper_method is EarlyStopperMethod.SSIM_TYPE):
            self.loss_op = kornia.losses.SSIMLoss(5)

        self.test_metric_list = []

    def update_last_metric(self, last_metric):
        self.last_metric = last_metric
        print("Updated last metric to: ", self.last_metric)

    def register_metric(self, input, target, epoch):
        if(epoch >= self.min_epochs):
            self.test_metric_list.append(EarlyStopper.TestMetric(input, target))

    def test(self, epoch):
        if(epoch < self.min_epochs):
            self.test_metric_list.clear()
            self.stop_counter = -1
            return False

        if(len(self.test_metric_list) == 0):
            print("No registered metric. Register a metric first.")
            return False

        with torch.no_grad(), amp.autocast():
            ave_D_loss = 0.0
            for test_metric in self.test_metric_list:
                input_tensor = test_metric.get_input()
                target_tensor = test_metric.get_target()
                ave_D_loss = ave_D_loss + self.loss_op(input_tensor, target_tensor)

            ave_D_loss = ave_D_loss / len(self.test_metric_list) * 1.0
            self.test_metric_list.clear()

        # if(self.last_metric < ave_D_loss):
        #     self.stop_counter += 1
        self.stop_counter += 1

        if(self.last_metric > ave_D_loss):
            self.last_metric = ave_D_loss
            self.stop_counter = 0
            print("Early stopping mechanism reset. Best metric is now ", self.last_metric.item())
            # trainer.save_states(epoch, iteration, self.last_metric)

        if (self.stop_counter == self.early_stop_tolerance):
            self.stop_condition_met = True
            print("Met stopping condition with best metric of: ", self.last_metric.item(), ". Latest metric: ", ave_D_loss)

        return self.stop_condition_met

    def has_reset(self):
        return (self.stop_counter == 0)

    def did_stop_condition_met(self):
        return self.stop_condition_met

    def get_last_metric(self):
        return self.last_metric









