import glob
import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from loaders import dataset_loader
from model import iteration_table
from trainers import relighting_trainer
from trainers import early_stopper
from utils import tensor_utils
import constants
import cv2

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--g_lr', type=float, help="LR", default="0.0002")
parser.add_option('--d_lr', type=float, help="LR", default="0.0002")
parser.add_option('--batch_size', type=int, help="batch_size", default="128")
parser.add_option('--num_blocks', type=int)
parser.add_option('--net_config', type=int)
parser.add_option('--version_name', type=str, help="version_name")
parser.add_option('--input_path', type=str)
parser.add_option('--output_path', type=str)
parser.add_option('--img_size', type=int, default=(256, 256))

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (constants.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    img_list = glob.glob(opts.input_path + "*.jpg") + glob.glob(opts.input_path + "*.png")
    print("Images found: ", len(img_list))

    trainer = relighting_trainer.RelightingTrainer(device, opts)

    constants.ITERATION = str(opts.iteration)
    constants.RELIGHTING_VERSION = opts.version_name
    constants.RELIGHTING_CHECKPATH = 'checkpoint/' + constants.RELIGHTING_VERSION + "_" + constants.ITERATION + '.pt'
    checkpoint = torch.load(constants.RELIGHTING_CHECKPATH, map_location=device)
    trainer.load_saved_state(checkpoint)

    normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for i, input_path in enumerate(img_list, 0):
        filename = input_path.split("\\")[-1]
        input_tensor = tensor_utils.load_metric_compatible_img(input_path, cv2.COLOR_BGR2RGB, True, True, opts.img_size).to(device)
        input_tensor = normalize_op(input_tensor)

        albedo_tensor = trainer.infer_albedo(input_tensor)
        shading_tensor = trainer.infer_shading(input_tensor)
        print(np.shape(albedo_tensor), np.shape(shading_tensor))

        albedo_tensor = albedo_tensor * 0.5 + 0.5

        vutils.save_image(albedo_tensor.squeeze(), opts.output_path + filename)




if __name__ == "__main__":
    main(sys.argv)
