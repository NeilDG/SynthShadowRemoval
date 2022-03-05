import random
import sys
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import constants
import kornia
from loaders import dataset_loader
from trainers import early_stopper
from trainers import shadow_relight_trainer
from model import iteration_table
from utils import plot_utils

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--bce_weight', type=float, help="Weight", default="0.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--mode', type=str, default="azimuth")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)
# parser.add_option('--input_light_angle', type=int, default="0")
# parser.add_option('--desired_light_angle', type=int, default="144")

# --img_to_load=-1 --load_previous=1
# Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.SHADOWMAP_RELIGHT_VERSION = opts.version_name
    constants.SHADOWMAP_RELIGHT_CHECKPATH = 'checkpoint/' + constants.SHADOWMAP_RELIGHT_VERSION + "_" + constants.ITERATION + '.pt'

    # COARE
    if (constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_PREFIX_5_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 5/"
        constants.DATASET_ALBEDO_5_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 5/albedo/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.SHADOWMAP_RELIGHT_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

    # GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.SHADOWMAP_RELIGHT_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_PREFIX_5_PATH = "/home/neil_delgallego/SynthWeather Dataset 5/"
        constants.DATASET_ALBEDO_5_PATH = "/home/neil_delgallego/SynthWeather Dataset 5/albedo/"


def show_images(img_tensor, caption):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    plt.figure(figsize=(32, 32))
    plt.axis("off")
    plt.title(caption)
    plt.imshow(np.transpose(vutils.make_grid(img_tensor.to(device)[:16], nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
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

    sample_path = constants.DATASET_PREFIX_5_PATH + opts.mode + "/" + "0deg/" + "shadow_map/"
    albedo_dir = constants.DATASET_ALBEDO_5_PATH
    shading_dir = constants.DATASET_PREFIX_5_PATH + "shading/"
    rgb_dir = constants.DATASET_PREFIX_5_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "rgb/"
    shadow_dir = constants.DATASET_PREFIX_5_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "shadow_map/"

    print(rgb_dir, albedo_dir, shading_dir, shadow_dir)

    # Create the dataloader
    train_loader = dataset_loader.load_map_train_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts)
    test_loader = dataset_loader.load_map_test_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts)
    # sp_train_loader = dataset_loader.load_shadow_priors_train(constants.DATASET_PREFIX_5_PATH + opts.mode + "/", opts)
    # sp_test_loader = dataset_loader.load_shadow_priors_test(constants.DATASET_PREFIX_5_PATH + opts.mode + "/", opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)
    start_epoch = 0
    iteration = 0

    # Plot some training images
    # if (constants.server_config == 0):
    #     _, rgb_batch, a_batch, b_batch, light_angle_batch = next(iter(train_loader))
    #
    #     show_images(a_batch, "Training - A Images")
    #     show_images(b_batch, "Training - B Images")

        # _, a_batch = next(iter(sp_train_loader))
        # show_images(a_batch, "Training - Shadow Priors")

    it_table = iteration_table.IterationTable()
    trainer = shadow_relight_trainer.ShadowRelightTrainerRGB(device, opts, it_table.is_bce_enabled(opts.iteration))
    trainer.update_penalties(opts.adv_weight, it_table.get_l1_weight(opts.iteration), it_table.get_lpip_weight(opts.iteration),
                             it_table.get_ssim_weight(opts.iteration), opts.bce_weight)

    last_metric = 10000.0
    if (opts.load_previous):
        checkpoint = torch.load(constants.SHADOWMAP_RELIGHT_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        last_metric = checkpoint[constants.LAST_METRIC_KEY]
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.SHADOWMAP_RELIGHT_CHECKPATH, start_epoch))
        print("===================================================")

    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000, last_metric)

    if (opts.test_mode == 1):
        print("Plotting test images...")
        _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(train_loader))
        input_rgb_tensor = input_rgb_batch.to(device)
        target_rgb_tensor = target_rgb_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        shading_tensor = shading_batch.to(device)
        input_shadow_tensor = input_shadow_batch.to(device)
        target_shadow_tensor = target_shadow_batch.to(device)
        light_angle_tensor = light_angle_batch.to(device)

        trainer.train(input_rgb_tensor, input_shadow_tensor, light_angle_tensor, albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor)
        trainer.visdom_visualize(input_rgb_tensor, input_shadow_tensor, light_angle_tensor, albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor, "Training")

        _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(test_loader))
        input_rgb_tensor = input_rgb_batch.to(device)
        target_rgb_tensor = target_rgb_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        shading_tensor = shading_batch.to(device)
        input_shadow_tensor = input_shadow_batch.to(device)
        target_shadow_tensor = target_shadow_batch.to(device)
        light_angle_tensor = light_angle_batch.to(device)

        trainer.visdom_visualize(input_rgb_tensor, input_shadow_tensor, light_angle_tensor, albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor, "Test")

        #plot metrics
        shadow_relight_tensor = (trainer.test(input_shadow_tensor, input_rgb_tensor, light_angle_tensor) * 0.5) + 0.5
        target_shadow_tensor = (target_shadow_tensor * 0.5) + 0.5

        psnr_relight = np.round(kornia.losses.psnr(shadow_relight_tensor, target_shadow_tensor, max_val=1.0).item(), 4)
        ssim_relight = np.round(1.0 - kornia.losses.ssim_loss(shadow_relight_tensor, target_shadow_tensor, 5).item(), 4)

        display_text = "Versions: " + opts.version_name + str(opts.iteration) + \
                       "<br> PSNR: " + str(psnr_relight) + "<br> SSIM: " +str(ssim_relight)

        visdom_reporter = plot_utils.VisdomReporter()
        visdom_reporter.plot_text(display_text)

    else:
        print("Starting Training Loop...")
        for i in range(0, 5):
            for epoch in range(start_epoch, constants.num_epochs):
                # For each batch in the dataloader
                for i, (train_data, test_data, rw_data) in enumerate(zip(train_loader, test_loader, rw_loader)):
                    _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = train_data
                    input_rgb_tensor = input_rgb_batch.to(device)
                    target_rgb_tensor = target_rgb_batch.to(device)
                    albedo_tensor = albedo_batch.to(device)
                    shading_tensor = shading_batch.to(device)
                    input_shadow_tensor = input_shadow_batch.to(device)
                    target_shadow_tensor = target_shadow_batch.to(device)
                    light_angle_tensor = light_angle_batch.to(device)

                    trainer.train(input_rgb_tensor, input_shadow_tensor, light_angle_tensor, albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor)
                    iteration = iteration + 1

                    stopper_method.test(trainer, epoch, iteration, trainer.test(input_shadow_tensor, input_rgb_tensor, light_angle_tensor), target_shadow_tensor)

                    if (stopper_method.did_stop_condition_met()):
                        break

                trainer.save_states_checkpt(epoch, iteration, stopper_method.get_last_metric())
                _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(test_loader))
                input_rgb_tensor = input_rgb_batch.to(device)
                target_rgb_tensor = target_rgb_batch.to(device)
                albedo_tensor = albedo_batch.to(device)
                shading_tensor = shading_batch.to(device)
                input_shadow_tensor = input_shadow_batch.to(device)
                target_shadow_tensor = target_shadow_batch.to(device)
                light_angle_tensor = light_angle_batch.to(device)

                trainer.visdom_visualize(input_rgb_tensor, input_shadow_tensor, light_angle_tensor, albedo_tensor, shading_tensor, target_shadow_tensor, target_rgb_tensor, "Test")
                trainer.visdom_plot(iteration)
                # _, rw_batch = rw_data
                # rw_tensor = rw_batch.to(device)
                # trainer.visdom_infer(rw_tensor)

                if (stopper_method.did_stop_condition_met()):
                    # visualize last result
                    # view_batch, test_a_batch, test_b_batch = test_data
                    # test_a_tensor = test_a_batch.to(device)
                    # test_b_tensor = test_b_batch.to(device)

                    # trainer.visdom_visualize(a_tensor, b_tensor, test_a_tensor, test_b_tensor)
                    break


if __name__ == "__main__":
    main(sys.argv)
