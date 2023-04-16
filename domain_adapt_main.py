import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader
from trainers import iid_trainer
from trainers import early_stopper
from utils import tensor_utils
import global_config
from trainers import domain_adapt_trainer
from trainers import domain_adapt_trainer_2

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--net_config', type=int)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)

# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.STYLE_TRANSFER_VERSION = opts.version_name
    constants.STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + constants.STYLE_TRANSFER_VERSION + "_" + constants.ITERATION + '.pt'

    # COARE
    if (constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.imgx_dir = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 6/azimuth/*/rgb/*.png"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)

    # GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)
        constants.imgx_dir = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.imgy_dir = "/home/neil_delgallego/SynthWeather Dataset 6/azimuth/*/rgb/*.png"


def main(argv):
    (opts, args) = parser.parse_args(argv)
    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    manualSeed = random.randint(1, 10000)  # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    train_loader = dataset_loader.load_da_dataset_train(constants.imgx_dir, constants.imgy_dir, opts)
    test_loader = dataset_loader.load_da_dataset_test(constants.imgx_dir, constants.imgy_dir, opts)

    index = 0
    start_epoch = 0
    iteration = 0

    # trainer = domain_adapt_trainer.DomainAdaptTrainer(device, opts)
    trainer = domain_adapt_trainer_2.DomainAdaptTrainer(device, opts)
    last_metric = 10000.0
    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000, last_metric)
    if (opts.load_previous):
        checkpoint = torch.load(constants.STYLE_TRANSFER_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        last_metric = checkpoint[constants.LAST_METRIC_KEY]
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.STYLE_TRANSFER_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        print("Plotting test images...")
        imgx_batch, imgy_batch = next(iter(train_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        trainer.train(imgx_tensor, imgy_tensor)
        trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Training")

        imgx_batch, imgy_batch = next(iter(test_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        trainer.train(imgx_tensor, imgy_tensor)
        trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Test")

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                imgx_batch, imgy_batch = train_data
                imgx_tensor = imgx_batch.to(device)
                imgy_tensor = imgy_batch.to(device)

                trainer.train(imgx_tensor, imgy_tensor)
                iteration = iteration + 1

                x2y, _ = trainer.test(imgx_tensor, imgy_tensor)
                stopper_method.test(trainer, epoch, iteration, x2y, imgy_tensor) #stop training if reconstruction no longer becomes close to Y

                if(i % 900 == 0):
                    trainer.save_states_checkpt(epoch, iteration, last_metric)
                    imgx_batch, imgy_batch = test_data
                    imgx_tensor = imgx_batch.to(device)
                    imgy_tensor = imgy_batch.to(device)

                    trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Test")
                    trainer.visdom_plot(iteration)

                if (stopper_method.did_stop_condition_met()):
                    break

            if (stopper_method.did_stop_condition_met()):
                break

if __name__ == "__main__":
    main(sys.argv)