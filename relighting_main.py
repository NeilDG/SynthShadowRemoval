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
from trainers import relighting_trainer
from trainers import early_stopper
from utils import tensor_utils
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--rgb_l1_weight', type=float, help="Weight", default="1.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--patch_size', type=int, help="patch_size", default="64")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--mode', type=str, default="elevation")
parser.add_option('--test_mode', type=int, help="Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help="Min epochs", default=120)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.RELIGHTING_VERSION = opts.version_name
    constants.RELIGHTING_CHECKPATH = 'checkpoint/' + constants.RELIGHTING_VERSION + "_" + constants.ITERATION + '.pt'

    # COARE
    if (constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_PREFIX_6_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 6/"
        constants.DATASET_ALBEDO_6_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 6/albedo/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.RELIGHTING_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

    # GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.RELIGHTING_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_PREFIX_6_PATH = "/home/neil_delgallego/SynthWeather Dataset 6/"
        constants.DATASET_ALBEDO_6_PATH = "/home/neil_delgallego/SynthWeather Dataset 6/albedo/"

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

    albedo_dir = constants.DATASET_ALBEDO_6_PATH
    shading_dir = constants.DATASET_PREFIX_6_PATH + "shading/"
    rgb_dir = constants.DATASET_PREFIX_6_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "rgb/"
    shadow_dir = constants.DATASET_PREFIX_6_PATH + opts.mode + "/" + "{input_light_angle}deg/" + "shadow_map/"

    print(rgb_dir, albedo_dir, shading_dir, shadow_dir)

    # Create the dataloader
    train_loader = dataset_loader.load_map_train_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts)
    test_loader = dataset_loader.load_map_test_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)

    GTA_BASE_PATH = "E:/IID-TestDataset/GTA/"
    RGB_PATH = GTA_BASE_PATH + "/input/"
    ALBEDO_PATH = GTA_BASE_PATH + "/albedo_white/"
    gta_loader = dataset_loader.load_gta_dataset(RGB_PATH, ALBEDO_PATH, opts)

    index = 0
    start_epoch = 0
    iteration = 0

    # Plot some training images
    # if (constants.server_config == 0):
    #     _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(train_loader))
    #     show_images(input_rgb_batch, "Training - A Images")
    #     show_images(albedo_batch, "Training - B Images")
    #     show_images(shading_batch, "Training - C Images")
    #     show_images(input_shadow_batch, "Training - D Images")
    #     show_images(target_shadow_batch, "Training - E Images")
    #     show_images(target_rgb_batch, "Training - F Images")

        # print(light_angle_batch)


    trainer = relighting_trainer.RelightingTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.rgb_l1_weight)

    last_metric = 10000.0
    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000, last_metric)
    if (opts.load_previous):
        checkpoint = torch.load(constants.RELIGHTING_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        last_metric = checkpoint[constants.LAST_METRIC_KEY]
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.RELIGHTING_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        print("Plotting test images...")
        _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(train_loader))
        input_rgb_tensor = input_rgb_batch.to(device)
        target_rgb_tensor = target_rgb_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        shading_tensor = shading_batch.to(device)
        input_shadow_tensor = input_shadow_batch.to(device)
        target_shadow_tensor = target_shadow_batch.to(device)
        light_angle_tensor = light_angle_batch.to(device)

        trainer.train(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor)
        # trainer.visdom_visualize(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor, "Training")

        _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = next(iter(test_loader))
        input_rgb_tensor = input_rgb_batch.to(device)
        target_rgb_tensor = target_rgb_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        shading_tensor = shading_batch.to(device)
        input_shadow_tensor = input_shadow_batch.to(device)
        target_shadow_tensor = target_shadow_batch.to(device)
        light_angle_tensor = light_angle_batch.to(device)

        trainer.visdom_visualize(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor, "Test")
        trainer.visdom_measure(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor, "Test")

        _, input_rgb_batch = next(iter(rw_loader))
        input_rgb_tensor = input_rgb_batch.to(device)
        trainer.visdom_infer(input_rgb_tensor)

        gta_rgb, gta_albedo = next(iter(gta_loader))
        gta_rgb_tensor = gta_rgb.to(device)
        trainer.visdom_measure_gta(gta_rgb_tensor, gta_albedo)


    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = train_data
                input_rgb_tensor = input_rgb_batch.to(device)
                target_rgb_tensor = target_rgb_batch.to(device)
                albedo_tensor = albedo_batch.to(device)
                shading_tensor = shading_batch.to(device)
                input_shadow_tensor = input_shadow_batch.to(device)
                target_shadow_tensor = target_shadow_batch.to(device)
                light_angle_tensor = light_angle_batch.to(device)

                trainer.train(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor)
                iteration = iteration + 1

                stopper_method.test(trainer, epoch, iteration, trainer.test(input_rgb_tensor), input_rgb_tensor)

                if (i % 300 == 0):
                    trainer.save_states_checkpt(epoch, iteration, last_metric)
                    _, input_rgb_batch, albedo_batch, shading_batch, input_shadow_batch, target_shadow_batch, target_rgb_batch, light_angle_batch = test_data
                    input_rgb_tensor = input_rgb_batch.to(device)
                    target_rgb_tensor = target_rgb_batch.to(device)
                    albedo_tensor = albedo_batch.to(device)
                    shading_tensor = shading_batch.to(device)
                    input_shadow_tensor = input_shadow_batch.to(device)
                    target_shadow_tensor = target_shadow_batch.to(device)
                    light_angle_tensor = light_angle_batch.to(device)
                    trainer.visdom_visualize(input_rgb_tensor, albedo_tensor, shading_tensor, input_shadow_tensor, input_rgb_tensor, "Test")
                    trainer.visdom_plot(iteration)

                if (stopper_method.did_stop_condition_met()):
                    break

            if (stopper_method.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)