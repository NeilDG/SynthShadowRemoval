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
from trainers import embedding_trainer
import constants
from trainers import early_stopper

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="10.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--use_bce', type=int, default = "0")
parser.add_option('--use_lpips', type=int, default = "0")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00002")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help= "Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--debug_mode', type=int, default=0)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.EMBEDDING_VERSION = opts.version_name
    constants.EMBEDDING_CHECKPATH = 'checkpoint/' + constants.EMBEDDING_VERSION + "_" + constants.ITERATION + '.pt'
    constants.plot_enabled = opts.plot_enabled

    if (opts.debug_mode == 1):
        constants.early_stop_threshold = 0
        constants.min_epochs = 1
        constants.num_epochs = 1

    if(constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_WEATHER_DEFAULT_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/default/"
        constants.DATASET_WEATHER_STYLED_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/styled/"
        constants.DATASET_WEATHER_SUNNY_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/sunny/"
        constants.DATASET_WEATHER_NIGHT_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/night/"
        constants.DATASET_WEATHER_CLOUDY_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/cloudy/"
        constants.DATASET_WEATHER_SEGMENT_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/segmentation/"

    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.EMBEDDING_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:16], nrow = 8, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

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

    constants.imgx_dir = "E:/Places Dataset/*.jpg"
    constants.imgy_dir = "E:/SynthWeather Dataset 7/azimuth/*/rgb/*.png"
    constants.imgx_dir_test = "E:/Places Dataset/*.jpg"
    constants.imgy_dir_test = "E:/SynthWeather Dataset 7/azimuth/*/rgb/*.png"

    train_loader = dataset_loader.load_ffa_dataset_train(constants.imgx_dir, constants.imgy_dir, opts)
    test_loader = dataset_loader.load_ffa_dataset_test(constants.imgx_dir_test, constants.imgy_dir_test, opts)

    trainer = embedding_trainer.EmbeddingTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.likeness_weight)
    start_epoch = 0
    iteration = 0
    last_metric = 10000.0
    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000, last_metric)

    if (opts.load_previous):
        checkpoint = torch.load(constants.EMBEDDING_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.EMBEDDING_CHECKPATH, start_epoch))
        print("===================================================")

    if (opts.test_mode == 1):
        print("Plotting test images...")
        imgx_batch, imgy_batch = next(iter(train_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Train")

        imgx_batch, imgy_batch = next(iter(test_loader))
        imgx_tensor = imgx_batch.to(device)
        imgy_tensor = imgy_batch.to(device)

        trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Test")

    else:
        print("Starting domain adaptation training loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                imgx_batch, imgy_batch = train_data
                imgx_tensor = imgx_batch.to(device)
                imgy_tensor = imgy_batch.to(device)

                trainer.train(imgx_tensor)
                trainer.train(imgy_tensor)

                iteration = iteration + 1

                imgx_batch, imgy_batch = test_data
                imgx_tensor = imgx_batch.to(device)
                imgy_tensor = imgy_batch.to(device)

                imgx_recon = trainer.test(imgx_tensor)
                stopper_method.test(trainer, epoch, iteration, imgx_recon, imgx_tensor)  # stop training if reconstruction no longer becomes close to X

                if (i % 256 == 0):
                    trainer.visdom_visualize(imgx_tensor, imgy_tensor, "Train")

                    trainer.save_states_checkpt(epoch, iteration)
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