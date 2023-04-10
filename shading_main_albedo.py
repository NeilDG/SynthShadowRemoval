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
from trainers import shading_trainer
from trainers import early_stopper
import global_config

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--l1_weight', type=float, help="Weight", default="10.0")
parser.add_option('--lpip_weight', type=float, help="Weight", default="0.0")
parser.add_option('--ssim_weight', type=float, help="Weight", default="0.0")
parser.add_option('--bce_weight', type=float, help="Weight", default="0.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--use_bce', type=int, default = "0")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--light_angle', type=int, help="Light angle", default = "0")
parser.add_option('--mode', type=str, default = "elevation")
parser.add_option('--test_mode', type=int, help= "Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help= "Min epochs", default=120)

#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.SHADING_VERSION = opts.version_name
    constants.SHADING_CHECKPATH = 'checkpoint/' + constants.SHADING_VERSION + "_" + constants.ITERATION + '.pt'

    # COARE
    if (constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_PREFIX_7_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 4/"
        constants.DATASET_ALBEDO_7_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 4/albedo/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.SHADOWMAP_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

    # GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.SHADOWMAP_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_PREFIX_7_PATH = "/home/neil_delgallego/SynthWeather Dataset 4/"
        constants.DATASET_ALBEDO_7_PATH = "/home/neil_delgallego/SynthWeather Dataset 4/albedo/"

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

    manualSeed = random.randint(1, 10000)  # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    albedo_path = constants.DATASET_ALBEDO_7_PATH
    #rgb_path = constants.DATASET_PREFIX_4_PATH + opts.mode + "/" + str(opts.light_angle) + "deg/" + "rgb/"
    rgb_path = constants.DATASET_PREFIX_7_PATH + "no_shadows/"
    map_path = constants.DATASET_PREFIX_7_PATH + "shading/"

    # Create the dataloader
    print(rgb_path, map_path)
    train_loader = dataset_loader.load_shadowmap_train_dataset(rgb_path, albedo_path, map_path, opts)
    test_loader = dataset_loader.load_shadowmap_test_dataset(rgb_path, albedo_path, map_path, opts)
    # rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)
    start_epoch = 0
    iteration = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a_batch, b_batch, c_batch = next(iter(train_loader))
        a_tensor = a_batch.to(device)
        b_tensor = b_batch.to(device)
        c_tensor = c_batch.to(device)

        show_images(a_tensor, "Training - A Images")
        show_images(b_tensor, "Training - B Images")
        show_images(c_tensor, "Training - C Images")

    trainer = shading_trainer.ShadingTrainerAlbedo(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.l1_weight, opts.lpip_weight, opts.ssim_weight, opts.bce_weight)

    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000)

    if (opts.load_previous):
        checkpoint = torch.load(constants.SHADING_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.SHADING_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        _, a_batch, b_batch, c_batch = next(iter(train_loader))
        a_tensor = a_batch.to(device)
        b_tensor = b_batch.to(device)
        c_tensor = c_batch.to(device)

        trainer.train(a_tensor, b_tensor, c_tensor)

        view_batch, test_a_batch, test_b_batch, test_c_batch = next(iter(test_loader))
        test_a_tensor = test_a_batch.to(device)
        test_b_tensor = test_b_batch.to(device)
        test_c_tensor = test_c_batch.to(device)
        trainer.visdom_visualize(a_tensor, b_tensor, c_tensor, test_a_tensor, test_b_tensor, test_c_tensor)

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, a_batch, b_batch, c_batch = train_data
                a_tensor = a_batch.to(device)
                b_tensor = b_batch.to(device)
                c_tensor = c_batch.to(device)

                trainer.train(a_tensor, b_tensor, c_tensor)
                iteration = iteration + 1

                stopper_method.test(trainer, epoch, iteration, trainer.test(a_tensor, c_tensor), b_tensor)

                if (stopper_method.did_stop_condition_met()):
                    break

            trainer.save_states_checkpt(epoch, iteration)
            view_batch, test_a_batch, test_b_batch, test_c_batch = test_data
            test_a_tensor = test_a_batch.to(device)
            test_b_tensor = test_b_batch.to(device)
            test_c_tensor = test_c_batch.to(device)
            trainer.visdom_visualize(a_tensor, b_tensor, c_tensor, test_a_tensor, test_b_tensor, test_c_tensor)

            if (stopper_method.did_stop_condition_met()):
                # visualize last result
                view_batch, test_a_batch, test_b_batch, test_c_batch = next(iter(test_loader))
                test_a_tensor = test_a_batch.to(device)
                test_b_tensor = test_b_batch.to(device)
                test_c_tensor = test_c_batch.to(device)
                trainer.visdom_visualize(a_tensor, b_tensor, c_tensor, test_a_tensor, test_b_tensor, test_c_tensor)
                break



if __name__ == "__main__":
    main(sys.argv)