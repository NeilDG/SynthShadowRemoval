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
from trainers import transfer_trainer
import global_config

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--load_previous', type=int, help="Load previous?", default=0)
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--identity_weight', type=float, help="Weight", default="0.0")
parser.add_option('--adv_weight', type=float, help="Weight", default="1.0")
parser.add_option('--likeness_weight', type=float, help="Weight", default="0.0")
parser.add_option('--smoothness_weight', type=float, help="Weight", default="0.0")
parser.add_option('--cycle_weight', type=float, help="Weight", default="10.0")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--use_bce', type=int)
parser.add_option('--g_lr', type=float, help="LR", default="0.00002")
parser.add_option('--d_lr', type=float, help="LR", default="0.00005")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--weather', type=str, help="Weather choice", default = "sunny")

#--img_to_load=-1 --load_previous=1
#Update config if on COARE
def update_config(opts):
    constants.server_config = opts.server_config
    constants.ITERATION = str(opts.iteration)
    constants.STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + constants.STYLE_TRANSFER_VERSION + "_" + constants.ITERATION + '.pt'

    if(constants.server_config == 1):
        print("Using COARE configuration ", opts.version_name)
        constants.STYLE_TRANSFER_VERSION = opts.version_name
        constants.STYLE_TRANSFER_CHECKPATH = 'checkpoint/' + constants.STYLE_TRANSFER_VERSION + "_" + constants.ITERATION + '.pt'
        constants.DATASET_PLACES_PATH = "/scratch1/scratch2/neil.delgallego/Places Dataset/"
        constants.DATASET_WEATHER_SUNNY_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/sunny/"
        constants.DATASET_WEATHER_NIGHT_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/night/"
        constants.DATASET_WEATHER_CLOUDY_PATH = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset/cloudy/"

    elif (constants.server_config == 2):
        print("Using CCS configuration. Workers: ", opts.num_workers, "Path: ", constants.STYLE_TRANSFER_CHECKPATH)
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

    if(opts.weather == "sunny"):
        weather_path = constants.DATASET_WEATHER_SUNNY_PATH
    elif(opts.weather == "night"):
        weather_path = constants.DATASET_WEATHER_NIGHT_PATH
    elif(opts.weather == "cloudy"):
        weather_path = constants.DATASET_WEATHER_CLOUDY_PATH
    else:
        print("Cannot determine weather choice. Defaulting to sunny")
        weather_path = constants.DATASET_WEATHER_SUNNY_PATH

    # Create the dataloader
    train_loader = dataset_loader.load_color_train_dataset(constants.DATASET_PLACES_PATH, weather_path, opts)
    test_loader = dataset_loader.load_color_test_dataset(constants.DATASET_PLACES_PATH, weather_path, opts)

    index = 0
    start_epoch = 0
    iteration = 0

    # Plot some training images
    if (constants.server_config == 0):
        _, a_batch, b_batch = next(iter(train_loader))

        show_images(a_batch, "Training - A Images")
        show_images(b_batch, "Training - B Images")

    trainer = transfer_trainer.TransferTrainer(device, opts)
    trainer.update_penalties(opts.adv_weight, opts.identity_weight, opts.l1_weight, opts.cycle_weight, opts.smoothness_weight)

    if (opts.load_previous):
        checkpoint = torch.load(constants.STYLE_TRANSFER_CHECKPATH, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration'] + 1
        trainer.load_saved_state(checkpoint)

        print("Loaded checkpt: %s Current epoch: %d" % (constants.STYLE_TRANSFER_CHECKPATH, start_epoch))
        print("===================================================")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, constants.num_epochs):
        # For each batch in the dataloader
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            _, a_batch, b_batch = train_data
            a_tensor = a_batch.to(device)
            b_tensor = b_batch.to(device)

            trainer.train(a_tensor, b_tensor)
            if (i % 100 == 0):
                trainer.save_states(epoch, iteration)

                view_batch, test_a_batch, test_b_batch = next(iter(test_loader))
                test_a_tensor =  test_a_batch.to(device)
                test_b_tensor = test_b_batch.to(device)
                trainer.visdom_report(iteration, a_tensor, b_tensor, test_a_tensor, test_b_tensor)

                iteration = iteration + 1


if __name__ == "__main__":
    main(sys.argv)