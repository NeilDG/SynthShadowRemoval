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
from trainers import iid_trainer, paired_trainer
from trainers import early_stopper
from transforms import iid_transforms
from utils import tensor_utils
import global_config
from trainers import embedding_trainer

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--rgb_l1_weight', type=float, help="Weight", default="1.0")
parser.add_option('--use_bce', type=int, default=0)
parser.add_option('--use_mask', type=int, default=0)
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--batch_size', type=int, help="batch_size", default="256")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=50)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.plot_enabled = opts.plot_enabled
    constants.min_epochs = opts.min_epochs
    constants.UNLIT_VERSION = opts.version_name
    constants.UNLIT_TRANSFER_CHECKPATH = 'checkpoint/' + constants.UNLIT_VERSION + "_" + constants.ITERATION + '.pt'

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers, " ", opts.version_name)


    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6


        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", opts.version_name)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8


    elif (constants.server_config == 4):
        opts.num_workers = 6


        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers, " ", opts.version_name)
    else:
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.imgx_dir = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.imgy_dir = "E:/SynthWeather Dataset 8/unlit/"
        print("Using HOME RTX3090 configuration. Workers: ", opts.num_workers, " ", opts.version_name)

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
    # torch.multiprocessing.set_sharing_strategy('file_system')
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    # manualSeed = random.randint(1, 10000)  # use if you want new results
    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    print(constants.imgx_dir, constants.imgy_dir)

    # Create the dataloader
    train_loader = dataset_loader.load_unlit_dataset_train(constants.imgx_dir, constants.imgy_dir, opts)
    test_loader =dataset_loader.load_unlit_dataset_test(constants.imgx_dir, constants.imgy_dir, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    start_epoch = 0
    iteration = 0

    trainer = paired_trainer.PairedTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight)

    if (opts.load_previous):
        checkpoint = torch.load(constants.UNLIT_TRANSFER_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.UNLIT_TRANSFER_CHECKPATH, start_epoch))
        print("===================================================")

    if (opts.test_mode == 1):
        print("Plotting test images...")
        _, rgb_batch, unlit_batch = next(iter(test_loader))
        rgb_tensor= rgb_batch.to(device)
        unlit_tensor = unlit_batch.to(device)

        trainer.visdom_visualize(rgb_tensor, unlit_tensor, torch.ones_like(rgb_tensor), "Test")

        _, rgb_ws_batch = next(iter(rw_loader))
        rgb_ws_tensor = rgb_ws_batch.to(device)
        trainer.visdom_infer(rgb_ws_tensor)


    else:
        print("Starting Training Loop...")
        last_metric = 10000.0
        stopper_method = early_stopper.EarlyStopper(constants.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, last_metric)
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, rgb_batch, unlit_batch = train_data
                rgb_tensor = rgb_batch.to(device)
                unlit_tensor = unlit_batch.to(device)

                trainer.train(rgb_tensor, unlit_tensor, torch.ones_like(rgb_tensor))

                iteration = iteration + 1
                stopper_method.register_metric(trainer.infer(rgb_tensor), unlit_tensor, epoch)
                stopper_method.test(trainer, epoch, iteration)
                if (stopper_method.did_stop_condition_met()):
                    break

                if(i % 300 == 0):
                    trainer.visdom_visualize(rgb_tensor, unlit_tensor, torch.ones_like(rgb_tensor), "Train")

                    _, rgb_batch, unlit_batch = test_data
                    rgb_tensor = rgb_batch.to(device)
                    unlit_tensor = unlit_batch.to(device)

                    trainer.visdom_visualize(rgb_tensor, unlit_tensor, torch.ones_like(rgb_tensor), "Test")
                    trainer.visdom_plot(iteration)
                    trainer.save_states_checkpt(epoch, iteration)

            if (stopper_method.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)