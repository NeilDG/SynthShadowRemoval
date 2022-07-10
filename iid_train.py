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
from transforms import iid_transforms
import constants
from trainers import embedding_trainer
from utils import plot_utils

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--rgb_l1_weight', type=float, help="Weight", default="1.0")
parser.add_option('--da_enabled', type=int, default=0)
parser.add_option('--da_version_name', type=str, default="")
parser.add_option('--albedo_mode', type=int, default="0")
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
parser.add_option('--debug_mode', type=int, default=0)
parser.add_option('--unlit_checkpt_file', type=str, default="")

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.IID_VERSION = opts.version_name
    constants.IID_CHECKPATH = 'checkpoint/' + constants.IID_VERSION + "_" + constants.ITERATION + '.pt'

    constants.plot_enabled = opts.plot_enabled

    if(opts.debug_mode == 1):
        constants.early_stop_threshold = 0
        constants.min_epochs = 1
        constants.num_epochs = 10
    else:
        constants.min_epochs = opts.min_epochs

    ## COARE
    if (constants.server_config == 1):
        opts.num_workers = 6
        print("Using COARE configuration. Workers: ", opts.num_workers, " ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 8/unlit/"

    # CCS JUPYTER
    elif (constants.server_config == 2):
        constants.num_workers = 6
        constants.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 8/albedo/"
        constants.DATASET_PLACES_PATH = constants.rgb_dir_ws

        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", opts.version_name)

    # GCLOUD
    elif (constants.server_config == 3):
        opts.num_workers = 8
        print("Using GCloud configuration. Workers: ", opts.num_workers, " ", opts.version_name)
        constants.DATASET_PLACES_PATH = "/home/neil_delgallego/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "/home/neil_delgallego/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "/home/neil_delgallego/SynthWeather Dataset 8/albedo/"

    elif (constants.server_config == 4):
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "D:/Datasets/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "D:/Datasets/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.albedo_dir = "D:/Datasets/SynthWeather Dataset 8/albedo/"

        print("Using HOME RTX2080Ti configuration. Workers: ", opts.num_workers, " ", opts.version_name)
    else:
        opts.num_workers = 6
        constants.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        constants.rgb_dir_ws = "E:/SynthWeather Dataset 8/train_rgb_styled/*/*.png"
        constants.rgb_dir_ns = "E:/SynthWeather Dataset 8/train_rgb_noshadows_styled/"
        constants.albedo_dir = "E:/SynthWeather Dataset 8/albedo/"
        constants.unlit_dir = "E:/SynthWeather Dataset 8/unlit/"
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

    print(constants.rgb_dir_ws, constants.albedo_dir)

    plot_utils.VisdomReporter.initialize()

    # Create the dataloader
    train_loader = dataset_loader.load_iid_datasetv2_train(constants.rgb_dir_ws, constants.rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, opts)
    test_loader = dataset_loader.load_iid_datasetv2_test(constants.rgb_dir_ws, constants.rgb_dir_ns, constants.unlit_dir, constants.albedo_dir, opts)
    rw_loader = dataset_loader.load_single_test_dataset(constants.DATASET_PLACES_PATH)

    start_epoch = 0
    iteration = 0

    trainer = iid_trainer.IIDTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.rgb_l1_weight)

    if (opts.load_previous):
        checkpoint = torch.load(constants.IID_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        last_metric = checkpoint[constants.LAST_METRIC_KEY]
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.IID_CHECKPATH, start_epoch))
        print("===================================================")

    if (opts.test_mode == 1):
        print("Plotting test images...")
        _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = next(iter(test_loader))
        rgb_ws_tensor = rgb_ws_batch.to(device)
        rgb_ns_tensor = rgb_ns_batch.to(device)
        albedo_tensor = albedo_batch.to(device)
        unlit_tensor = unlit_batch.to(device)
        iid_op = iid_transforms.IIDTransform()
        rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

        trainer.visdom_visualize(rgb_ws_tensor,unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Test")
        # trainer.visdom_measure(rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Test")

        _, rgb_ws_batch = next(iter(rw_loader))
        rgb_ws_tensor = rgb_ws_batch.to(device)
        trainer.visdom_infer(rgb_ws_tensor)

        GTA_BASE_PATH = "E:/IID-TestDataset/GTA/"
        RGB_PATH = GTA_BASE_PATH + "/input/"
        ALBEDO_PATH = GTA_BASE_PATH + "/albedo_white/"
        gta_loader = dataset_loader.load_gta_dataset(RGB_PATH, ALBEDO_PATH, opts)

        gta_rgb, gta_albedo = next(iter(gta_loader))
        gta_rgb_tensor = gta_rgb.to(device)
        trainer.visdom_measure_gta(gta_rgb_tensor, gta_albedo)


    else:
        print("Starting Training Loop...")
        iid_op = iid_transforms.IIDTransform().to(device)
        last_metric = 10000.0
        stopper_method_s = early_stopper.EarlyStopper(constants.min_epochs, early_stopper.EarlyStopperMethod.L1_TYPE, constants.early_stop_threshold, last_metric)
        for epoch in range(start_epoch, constants.num_epochs):
            # For each batch in the dataloader
            for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
                _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = train_data
                rgb_ws_tensor = rgb_ws_batch.to(device)
                rgb_ns_tensor = rgb_ns_batch.to(device)
                albedo_tensor = albedo_batch.to(device)
                unlit_tensor = unlit_batch.to(device)
                rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

                trainer.train(rgb_ws_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor)
                rgb2albedo, rgb2shading, rgb2shadow, _ = trainer.decompose(rgb_ws_tensor)

                iteration = iteration + 1
                stopper_method_s.register_metric(rgb2albedo, albedo_tensor, epoch)
                stopper_method_s.register_metric(rgb2shading, shading_tensor, epoch)
                stopper_method_s.register_metric(rgb2shadow, shadow_tensor, epoch)
                stopper_method_s.test(trainer, epoch, iteration)
                if (stopper_method_s.did_stop_condition_met()):
                    break

                if(i % 300 == 0):
                    trainer.visdom_visualize(rgb_ws_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Train")
                    _, rgb_ws_batch, rgb_ns_batch, albedo_batch, unlit_batch = test_data
                    rgb_ws_tensor = rgb_ws_batch.to(device)
                    rgb_ns_tensor = rgb_ns_batch.to(device)
                    albedo_tensor = albedo_batch.to(device)
                    unlit_tensor = unlit_batch.to(device)
                    rgb_ws_tensor, albedo_tensor, shading_tensor, shadow_tensor = iid_op(rgb_ws_tensor, rgb_ns_tensor, albedo_tensor)

                    trainer.visdom_visualize(rgb_ws_tensor, unlit_tensor, albedo_tensor, shading_tensor, shadow_tensor, "Test")
                    trainer.visdom_plot(iteration)
                    trainer.save_states_checkpt(epoch, iteration, last_metric)

            if (stopper_method_s.did_stop_condition_met()):
                break



if __name__ == "__main__":
    main(sys.argv)