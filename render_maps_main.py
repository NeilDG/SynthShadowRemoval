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
from trainers import render_maps_trainer
from trainers import early_stopper
import constants

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
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--use_bce', type=int, default = "0")
parser.add_option('--use_mask', type=int, default = "1")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--map_choice', type=str, help="Map choice", default = "albedo")
parser.add_option('--test_mode', type=int, help= "Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help= "Min epochs", default=60)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.MAPPER_VERSION = opts.version_name
    constants.MAPPER_CHECKPATH = 'checkpoint/' + constants.MAPPER_VERSION + "_" + constants.ITERATION + '.pt'

    #COARE
    if(constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_RGB_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 2/default/"
        constants.DATASET_ALBEDO_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 2/albedo/"
        constants.DATASET_NORMAL_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 2/normal/"
        constants.DATASET_SPECULAR_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 2/specular/"
        constants.DATASET_SMOOTHNESS_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 2/smoothness/"

    #CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.MAPPER_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

    #GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.MAPPER_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_RGB_PATH = "/home/neil_delgallego/SynthWeather Dataset/default/"
        constants.DATASET_ALBEDO_PATH = "/home/neil_delgallego/SynthWeather Dataset/albedo/"
        constants.DATASET_NORMAL_PATH = "/home/neil_delgallego/SynthWeather Dataset/normal/"
        constants.DATASET_SPECULAR_PATH = "/home/neil_delgallego/SynthWeather Dataset/specular/"
        constants.DATASET_SMOOTHNESS_PATH = "/home/neil_delgallego/SynthWeather Dataset/smoothness/"
        constants.DATASET_WEATHER_SEGMENT_PATH = "/home/neil_delgallego/SynthWeather Dataset/segmentation/"

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

    if(opts.map_choice == "albedo"):
        map_path = constants.DATASET_ALBEDO_PATH
    elif(opts.map_choice == "normal"):
        map_path = constants.DATASET_NORMAL_PATH
    elif(opts.map_choice == "specular"):
        map_path = constants.DATASET_SPECULAR_PATH
    elif (opts.map_choice == "smoothness"):
        map_path = constants.DATASET_SMOOTHNESS_PATH
    elif (opts.map_choice == "segmentation"):
        map_path = constants.DATASET_WEATHER_SEGMENT_PATH
    else:
        print("Cannot determine map choice. Defaulting to Albedo")
        map_path = constants.DATASET_ALBEDO_PATH

    # Create the dataloader
    train_loader = dataset_loader.load_map_train_dataset(constants.DATASET_RGB_PATH, map_path, opts)
    test_loader = dataset_loader.load_map_test_dataset(constants.DATASET_RGB_PATH, map_path, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)
    index = 0
    start_epoch = 0
    iteration = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a_batch, b_batch, mask_batch = next(iter(train_loader))

        show_images(a_batch, "Training - A Images")
        show_images(b_batch, "Training - B Images")
        # show_images(mask_batch, "Training - Mask Images")
        print(np.shape(mask_batch))
        print(mask_batch[0])

    trainer = render_maps_trainer.RenderMapsTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.l1_weight, opts.lpip_weight, opts.ssim_weight)

    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000)

    if (opts.load_previous):
        checkpoint = torch.load(constants.MAPPER_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.MAPPER_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        print("Plotting test images...")
        _, a_batch, b_batch, mask_batch = next(iter(train_loader))
        a_tensor = a_batch.to(device)
        b_tensor = b_batch.to(device)
        mask_tensor = mask_batch.to(device)

        trainer.train(a_tensor, b_tensor, mask_tensor)

        view_batch, test_a_batch, test_b_batch, test_mask_batch = next(iter(test_loader))
        test_a_tensor = test_a_batch.to(device)
        test_b_tensor = test_b_batch.to(device)
        test_mask_tensor = test_mask_batch.to(device)
        trainer.visdom_visualize(a_tensor, b_tensor, mask_tensor, test_a_tensor, test_b_tensor, test_mask_tensor)

        _, rw_batch = next(iter(rw_loader))
        rw_tensor = rw_batch.to(device)
        trainer.visdom_infer(rw_tensor)

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data, rw_data) in enumerate(zip(train_loader, test_loader, itertools.cycle(rw_loader))):
                _, a_batch, b_batch, mask_batch = train_data
                a_tensor = a_batch.to(device)
                b_tensor = b_batch.to(device)
                mask_tensor = mask_batch.to(device)
                trainer.train(a_tensor, b_tensor, mask_tensor)
                iteration = iteration + 1

                stopper_method.test(trainer, epoch, iteration, trainer.test(a_tensor, mask_tensor), b_tensor)

                if (i % 300 == 0):
                    trainer.save_states_checkpt(epoch, iteration)
                    view_batch, test_a_batch, test_b_batch, test_mask_batch = next(iter(test_loader))
                    test_a_tensor = test_a_batch.to(device)
                    test_b_tensor = test_b_batch.to(device)
                    test_mask_tensor = test_mask_batch.to(device)
                    trainer.visdom_plot(iteration)
                    trainer.visdom_visualize(a_tensor, b_tensor, mask_tensor, test_a_tensor, test_b_tensor, test_mask_tensor)

                    _, rw_batch = rw_data
                    rw_tensor = rw_batch.to(device)
                    trainer.visdom_infer(rw_tensor)

                if (stopper_method.did_stop_condition_met()):
                    break

            if (stopper_method.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)