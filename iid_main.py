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
import constants

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--net_config_a', type=int)
parser.add_option('--net_config_s', type=int)
parser.add_option('--num_blocks_a', type=int)
parser.add_option('--num_blocks_s', type=int)
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--l1_weight', type=float, help="Weight", default="10.0")
parser.add_option('--iid_weight', type=float, help="Weight", default="0.0")
parser.add_option('--lpip_weight', type=float, help="Weight", default="0.0")
parser.add_option('--ssim_weight', type=float, help="Weight", default="0.0")
parser.add_option('--bce_weight', type=float, help="Weight", default="0.0")
parser.add_option('--use_bce', type=int, default = "0")
parser.add_option('--use_mask', type=int, default = "1")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help= "Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help= "Min epochs", default=120)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.IID_VERSION = opts.version_name
    constants.IID_CHECKPATH = 'checkpoint/' + constants.IID_VERSION + "_" + constants.ITERATION + '.pt'

    #COARE
    if(constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_RGB_DECOMPOSE_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 3/rgb/"
        constants.DATASET_SHADING_DECOMPOSE_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 3/shading/"
        constants.DATASET_ALBEDO_DECOMPOSE_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 3/albedo/"

    #CCS JUPYTER
    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)
        constants.DATASET_PLACES_PATH = "Places Dataset/"

    #GCLOUD
    elif (constants.server_config == 3):
        print("Using GCloud configuration. Workers: ", opts.num_workers, "Path: ", constants.IID_CHECKPATH)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/"
        constants.DATASET_RGB_DECOMPOSE_PATH = "/home/neil_delgallego/SynthWeather Dataset 3/rgb/"
        constants.DATASET_SHADING_DECOMPOSE_PATH = "/home/neil_delgallego/SynthWeather Dataset 3/shading/"
        constants.DATASET_ALBEDO_DECOMPOSE_PATH = "/home/neil_delgallego/SynthWeather Dataset 3/albedo/"

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

    # Create the dataloader
    train_albedo_loader = dataset_loader.load_map_train_dataset(constants.DATASET_RGB_DECOMPOSE_PATH, constants.DATASET_ALBEDO_DECOMPOSE_PATH, opts)
    train_shading_loader = dataset_loader.load_map_train_dataset(constants.DATASET_RGB_DECOMPOSE_PATH, constants.DATASET_SHADING_DECOMPOSE_PATH, opts)
    test_albedo_loader = dataset_loader.load_map_test_dataset(constants.DATASET_RGB_DECOMPOSE_PATH, constants.DATASET_ALBEDO_DECOMPOSE_PATH, opts)
    test_shading_loader = dataset_loader.load_map_test_dataset(constants.DATASET_RGB_DECOMPOSE_PATH, constants.DATASET_SHADING_DECOMPOSE_PATH, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH, opts)
    start_epoch = 0
    iteration = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, rgb_batch, map_batch, mask_batch = next(iter(train_albedo_loader))

        show_images(rgb_batch, "Training - A Images")
        show_images(map_batch, "Training - B Images")
        show_images(mask_batch, "Training - Mask Images")

        _, rgb_batch, map_batch, mask_batch = next(iter(train_shading_loader))

        show_images(rgb_batch, "Training - A Images")
        show_images(map_batch, "Training - B Images")
        show_images(mask_batch, "Training - Mask Images")

    trainer = iid_trainer.IIDTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.l1_weight, opts.iid_weight, opts.lpip_weight, opts.ssim_weight, opts.bce_weight)

    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000)

    if (opts.load_previous):
        checkpoint = torch.load(constants.IID_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.IID_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        print("Plotting test images...")
        _, rgb_batch, map_batch, mask_batch = next(iter(train_albedo_loader))
        rgb_tensor = rgb_batch.to(device)
        albedo_tensor = map_batch.to(device)
        mask_tensor = mask_batch.to(device)

        _, rgb_batch, map_batch, _ = next(iter(train_shading_loader))
        rgb_tensor = rgb_batch.to(device)
        shading_tensor = map_batch.to(device)

        trainer.train(rgb_tensor, albedo_tensor, shading_tensor, mask_tensor)

        _, rgb_batch, map_batch, _ = next(iter(test_albedo_loader))
        rgb_tensor = rgb_batch.to(device)
        albedo_tensor = map_batch.to(device)

        _, _, map_batch, _ = next(iter(test_shading_loader))
        shading_tensor = map_batch.to(device)
        trainer.visdom_visualize(rgb_tensor, albedo_tensor, shading_tensor, True)

        _, rw_batch = next(iter(rw_loader))
        rw_tensor = rw_batch.to(device)
        trainer.visdom_infer(rw_tensor)

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_albedo_data, train_shading_data) in enumerate(zip(train_albedo_loader, train_shading_loader)):
                _, rgb_batch, map_batch, mask_batch = train_albedo_data
                rgb_tensor = rgb_batch.to(device)
                albedo_tensor = map_batch.to(device)
                mask_tensor = mask_batch.to(device)

                _, _, map_batch, _ = train_shading_data
                shading_tensor = map_batch.to(device)
                trainer.train(rgb_tensor, albedo_tensor, shading_tensor, mask_tensor)
                iteration = iteration + 1

                stopper_method.test(trainer, epoch, iteration, trainer.test(rgb_tensor), rgb_tensor)

                if (stopper_method.did_stop_condition_met()):
                    break

            trainer.visdom_plot(iteration)
            trainer.save_states_checkpt(epoch, iteration)

            _, rgb_batch, map_batch, _ = next(itertools.cycle(test_albedo_loader))
            rgb_tensor = rgb_batch.to(device)
            albedo_tensor = map_batch.to(device)

            _, _, map_batch, _ = next(itertools.cycle(test_shading_loader))
            shading_tensor = map_batch.to(device)
            trainer.visdom_visualize(rgb_tensor, albedo_tensor, shading_tensor, True)

            _, rw_batch = next(itertools.cycle(rw_loader))
            rw_tensor = rw_batch.to(device)
            trainer.visdom_infer(rw_tensor)

            if (stopper_method.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)