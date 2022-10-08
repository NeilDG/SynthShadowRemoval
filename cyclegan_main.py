# -*- coding: utf-8 -*-
"""
Main entry for GAN training
Created on Sun Apr 19 13:22:06 2020

@author: delgallegon
"""

from __future__ import print_function
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from config import iid_server_config
from loaders import dataset_loader
from trainers import cyclegan_trainer, early_stopper
import constants
from transforms import cyclegan_transforms, iid_transforms

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00005")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version', type=str, help="")
parser.add_option('--debug_run', type=int, help="Debug mode?", default=0)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled
    constants.debug_run = opts.debug_run

    # COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers, " ", opts.version)
        constants.imgx_dir = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 9/rgb/*/*.png"
        constants.imgx_dir_test = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir_test = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 9/rgb/*/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        opts.num_workers = 12
        print("Using CCS configuration. Workers: ", opts.num_workers, " ", opts.version)
        constants.imgx_dir = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 9/rgb/*/*.png"

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers, " ", opts.version)
        constants.imgx_dir = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/home/neil_delgallego/SynthWeather Dataset 9/rgb/*/*.png"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.imgx_dir = "C:/Datasets/Places Dataset/*.jpg"
        constants.imgy_dir = "C:/Datasets/SynthWeather Dataset 9/rgb/*/*.png"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers, " ", opts.version)
    else:
        opts.num_workers = 12
        constants.imgx_dir = "E:/Places Dataset/*.jpg"
        constants.imgy_dir = "E:/SynthWeather Dataset 10/synth_city/*/*/*.png"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers, " ", opts.version)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)
    
    # manualSeed = random.randint(1, 10000) # use if you want new results
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    constants.network_version = opts.version
    iid_server_config.IIDServerConfig.initialize()
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    general_config = sc_instance.get_general_configs()
    network_config = sc_instance.interpret_style_transfer_config_from_version()

    gt = cyclegan_trainer.CycleGANTrainer(device, opts)
    iteration = 0
    start_epoch = sc_instance.get_last_epoch_from_mode("train_style_transfer")
    
    # Create the dataloader
    train_loader = dataset_loader.load_da_dataset_train(constants.imgx_dir, constants.imgy_dir, opts)
    test_loader = dataset_loader.load_da_dataset_test(constants.imgx_dir, constants.imgy_dir, opts)

    print("Starting Training Loop...")

    for epoch in range(start_epoch, general_config["train_style_transfer"]["max_epochs"]):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            imgx_batch, imgy_batch = train_data
            imgx_tensor = imgx_batch.to(device)
            imgy_tensor = imgy_batch.to(device)

            imgx_ind = torch.randperm(len(imgx_tensor)) #shuffle both tensors constantly to ensure no pairing is learned.
            imgy_ind = torch.randperm(len(imgy_tensor))

            imgx_tensor = imgx_tensor[imgx_ind]
            imgy_tensor = imgy_tensor[imgy_ind]

            gt.train(epoch, iteration, imgx_tensor, imgy_tensor, iteration)
            iteration = iteration + 1

            if((i * network_config["load_size"]) % (network_config["batch_size"] * 100) == 0): #every X batches
                print("Iteration:", iteration)
                gt.visdom_visualize(imgx_tensor, imgy_tensor, "Train")

                gt.save_states(epoch, iteration, False)
                imgx_batch, imgy_batch = test_data
                imgx_tensor = imgx_batch.to(device)
                imgy_tensor = imgy_batch.to(device)

                gt.visdom_visualize(imgx_tensor, imgy_tensor, "Test")
                gt.visdom_plot(iteration)

            if (gt.is_stop_condition_met()):
                break

        if (gt.is_stop_condition_met()):
            break

#FIX for broken pipe num_workers issue.
if __name__=="__main__": 
    main(sys.argv)

