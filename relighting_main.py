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
parser.add_option('--l1_weight', type=float, help="Weight", default="10.0")
parser.add_option('--lpip_weight', type=float, help="Weight", default="0.0")
parser.add_option('--ssim_weight', type=float, help="Weight", default="0.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--use_bce', type=int, default = "0")
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--test_mode', type=int, help= "Test mode?", default=0)
parser.add_option('--min_epochs', type=int, help= "Min epochs", default=60)

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.RELIGHTING_VERSION = opts.version_name
    constants.RELIGHTING_CHECKPATH = 'checkpoint/' + constants.RELIGHTING_VERSION + "_" + constants.ITERATION + '.pt'

    #COARE
    if(constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_RGB_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/default/"
        constants.DATASET_ALBEDO_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/albedo/"
        constants.DATASET_NORMAL_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/normal/"
        constants.DATASET_SPECULAR_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/specular/"
        constants.DATASET_SMOOTHNESS_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/smoothness/"

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

    # Create the dataloader
    train_loader = dataset_loader.load_render_train_dataset(constants.DATASET_RGB_PATH, constants.DATASET_ALBEDO_PATH, constants.DATASET_NORMAL_PATH,
                                                            constants.DATASET_SPECULAR_PATH, constants.DATASET_SMOOTHNESS_PATH, constants.DATASET_LIGHTMAP_PATH, opts)

    test_loader = dataset_loader.load_render_test_dataset(constants.DATASET_RGB_PATH, constants.DATASET_ALBEDO_PATH, constants.DATASET_NORMAL_PATH,
                                                            constants.DATASET_SPECULAR_PATH, constants.DATASET_SMOOTHNESS_PATH, constants.DATASET_LIGHTMAP_PATH, opts)
    index = 0
    start_epoch = 0
    iteration = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a_batch, b_batch, c_batch, d_batch, e_batch, f_batch, = next(iter(train_loader))
        show_images(a_batch, "Training - A Images")
        show_images(b_batch, "Training - B Images")
        show_images(c_batch, "Training - C Images")
        show_images(d_batch, "Training - D Images")
        show_images(tensor_utils.interpret_one_hot(e_batch), "Training - E Images")
        show_images(f_batch, "Training - F Images")

    trainer = relighting_trainer.RelightingTrainer(device, 16, opts)
    trainer.update_penalties(opts.adv_weight, opts.l1_weight, opts.lpip_weight, opts.ssim_weight)

    stopper_method = early_stopper.EarlyStopper(opts.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, 2000)

    if (opts.load_previous):
        checkpoint = torch.load(constants.RELIGHTING_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.RELIGHTING_CHECKPATH, start_epoch))
        print("===================================================")

    if(opts.test_mode == 1):
        print("Plotting test images...")
        _, a_batch, b_batch, c_batch, d_batch, e_batch, f_batch, = next(iter(train_loader))
        a_tensor = a_batch.to(device)
        b_tensor = b_batch.to(device)
        c_tensor = c_batch.to(device)
        d_tensor = d_batch.to(device)
        e_tensor = e_batch.to(device)
        f_tensor = f_batch.to(device)

        trainer.train(torch.cat([b_tensor, c_tensor, d_tensor, e_tensor, f_tensor], 1), a_tensor)

        _, view_a_batch, view_b_batch, view_c_batch, view_d_batch, view_e_batch, view_f_batch, = next(iter(test_loader))
        view_a_tensor = view_a_batch.to(device)
        view_b_tensor = view_b_batch.to(device)
        view_c_tensor = view_c_batch.to(device)
        view_d_tensor = view_d_batch.to(device)
        view_e_tensor = view_e_batch.to(device)
        view_f_tensor = view_f_batch.to(device)
        trainer.visdom_visualize(b_tensor, torch.cat([b_tensor, c_tensor, d_tensor, e_tensor, f_tensor], 1), a_tensor,
                                 view_b_tensor, torch.cat([view_b_tensor, view_c_tensor, view_d_tensor, view_e_tensor, view_f_tensor], 1), view_a_tensor)

    else:
        print("Starting Training Loop...")
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, a_batch, b_batch, c_batch, d_batch, e_batch, f_batch, = train_data
                a_tensor = a_batch.to(device)
                b_tensor = b_batch.to(device)
                c_tensor = c_batch.to(device)
                d_tensor = d_batch.to(device)
                e_tensor = e_batch.to(device)
                f_tensor = f_batch.to(device)

                input_tensor = torch.cat([b_tensor, c_tensor, d_tensor, e_tensor, f_tensor], 1)
                trainer.train(input_tensor, a_tensor)
                iteration = iteration + 1

                stopper_method.test(trainer, epoch, iteration, trainer.test(input_tensor), a_tensor)

                if (i % 300 == 0):
                    trainer.save_states_checkpt(epoch, iteration)
                    _, view_a_batch, view_b_batch, view_c_batch, view_d_batch, view_e_batch, view_f_batch, = test_data
                    view_a_tensor = view_a_batch.to(device)
                    view_b_tensor = view_b_batch.to(device)
                    view_c_tensor = view_c_batch.to(device)
                    view_d_tensor = view_d_batch.to(device)
                    view_e_tensor = view_e_batch.to(device)
                    view_f_tensor = view_f_batch.to(device)
                    # trainer.visdom_visualize(torch.cat([b_tensor, c_tensor, d_tensor, e_tensor, f_tensor], 1), a_tensor,
                    #                          torch.cat([view_b_tensor, view_c_tensor, view_d_tensor, view_e_tensor, view_f_tensor], 1), view_a_tensor)
                    # trainer.visdom_plot(iteration)

                if (stopper_method.did_stop_condition_met()):
                    break

            if (stopper_method.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)