import sys
from optparse import OptionParser
import numpy as np
import kornia.color
from tqdm import tqdm

import global_config
from config import iid_server_config
from loaders import dataset_loader
import torch

from utils import tensor_utils

parser = OptionParser()
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--shadow_matte_network_version', type=str, default="v58.34")
parser.add_option('--shadow_removal_version', type=str, default="v58.28")
parser.add_option('--dataset_version_to_refine', type=str, default="v32_istd_styled")
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--num_workers', type=int, help="Workers", default="12")
parser.add_option('--train_mode', type=str, default="all") #all, train_shadow_matte, train_shadow

# Running mean of Synth dataset is:  tensor(7.9453e-05)  Std dev:  tensor(0.0002)
# Synth dataset min:  tensor(-0.0189) tensor(0.8824)

def quantify_datasets(opts):
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    dataset_version = opts.dataset_version_to_refine

    global_config.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    global_config.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
    rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)

    ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "E:/ISTD_Dataset/test/test_B/*.png"

    # load_size = 256
    # istd_loader, dataset_count = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, load_size, opts)
    #
    # needed_progress = int(dataset_count / load_size)
    # current_progress = 0
    # pbar = tqdm(total=needed_progress)
    # pbar.update(current_progress)
    #
    # istd_analyzer = DatasetAnalyzer()
    # istd_hsv_analyzer = DatasetHSVAnalyzer()
    # for i, (file_name, input_ws, input_ns, _, shadow_matte) in enumerate(istd_loader, 0):
    #     input_ws = input_ws.to(device)
    #     input_ns = input_ns.to(device)
    #     shadow_matte = shadow_matte.to(device)
    #     istd_analyzer.compute_std_mean_of_batch(shadow_matte)
    #     istd_hsv_analyzer.compute_std_mean_of_batch(input_ws)
    #     pbar.update(1)
    #
    # pbar.close()
    # print()
    #
    # istd_mean = istd_analyzer.get_std_mean()[1]
    # istd_std = istd_analyzer.get_std_mean()[0]
    # istd_mean_h = istd_hsv_analyzer.get_std_mean_h()[1]
    # istd_std_h = istd_hsv_analyzer.get_std_mean_h()[0]
    # istd_mean_s = istd_hsv_analyzer.get_std_mean_s()[1]
    # istd_std_s = istd_hsv_analyzer.get_std_mean_s()[0]
    # istd_mean_v = istd_hsv_analyzer.get_std_mean_v()[1]
    # istd_std_v = istd_hsv_analyzer.get_std_mean_v()[0]
    #
    #
    # print("Running mean of ISTD is: ", istd_mean, " Std dev: ", istd_std)
    # # print("ISTD min: ", istd_analyzer.get_min(), istd_analyzer.get_max())
    # print("-------")
    # print("Hue mean of ISTD is: ", istd_mean_h, " Std dev: ", istd_std_h)
    # print("Sat mean of ISTD is: ", istd_mean_s, " Std dev: ", istd_std_s)
    # print("Val mean of ISTD is: ", istd_mean_v, " Std dev: ", istd_std_v)
    # print("ISTD HSV range: ", istd_hsv_analyzer.get_min(), istd_hsv_analyzer.get_max())

    load_size = 128
    train_loader, dataset_count = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, 256, load_size, opts)

    needed_progress = int(dataset_count / load_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    synth_analyzer = DatasetAnalyzer()
    synth_hsv_analyzer = DatasetHSVAnalyzer()
    for i, (file_name, input_ws, input_ns, _, _, shadow_matte) in enumerate(train_loader, 0):
        input_ws = input_ws.to(device)
        input_ns = input_ns.to(device)
        shadow_matte = shadow_matte.to(device)
        synth_analyzer.compute_std_mean_of_batch(shadow_matte)
        synth_hsv_analyzer.compute_std_mean_of_batch(input_ws)
        pbar.update(1)

    pbar.close()
    print()

    synth_mean_h = synth_hsv_analyzer.get_std_mean_h()[1]
    synth_std_h = synth_hsv_analyzer.get_std_mean_h()[0]
    synth_mean_s = synth_hsv_analyzer.get_std_mean_s()[1]
    synth_std_s = synth_hsv_analyzer.get_std_mean_s()[0]
    synth_mean_v = synth_hsv_analyzer.get_std_mean_v()[1]
    synth_std_v = synth_hsv_analyzer.get_std_mean_v()[0]
    print("Running mean of Synth ", opts.dataset_version_to_refine, " dataset is: ", synth_analyzer.get_std_mean()[1], " Std dev: ", synth_analyzer.get_std_mean()[0])
    # print("Synth dataset min: ", synth_analyzer.get_min(), synth_analyzer.get_max())
    print("-------")
    print("Hue mean of Synth is: ", synth_mean_h, " Std dev: ", synth_std_h)
    print("Sat mean of Synth is: ", synth_mean_s, " Std dev: ", synth_std_s)
    print("Val mean of Synth is: ", synth_mean_v, " Std dev: ", synth_std_v)
    print("SYNTH HSV range: ", synth_hsv_analyzer.get_min(), synth_hsv_analyzer.get_max())

def prepare_clean(opts):
    global_config.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    global_config.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
    rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=opts.dataset_version_to_refine)
    rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=opts.dataset_version_to_refine)

    ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"

    ws_list = dataset_loader.assemble_img_list(rgb_dir_ws, opts)
    ns_list = dataset_loader.assemble_img_list(rgb_dir_ns, opts)

    istd_mean = -0.0003
    istd_std = 0.0345 * 2.0

    # print("Dataset len before: ", len(ws_list))
    # index = 27293
    # ws_list = ws_list[index: len(ws_list)]
    # ns_list = ns_list[index: len(ns_list)]

    print("Trimming dataset. Dataset len after: ", len(ws_list))

    dataset_loader.clean_dataset_using_std_mean(ws_list, ns_list, istd_mean, istd_std)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    global_config.shadow_removal_version = opts.shadow_removal_version
    global_config.shadow_matte_network_version = opts.shadow_matte_network_version
    iid_server_config.IIDServerConfig.initialize()

    # quantify_datasets(opts)
    prepare_clean(opts)


class DatasetAnalyzer():
    def __init__(self):
        print("Initializing data analyzer")
        self.running_std_mean = None
        self.batch = 0

        self.min = 100.0
        self.max = -100.0

    def compute_std_mean_of_batch(self, shadow_matte):
        shadow_matte = tensor_utils.normalize_to_01(shadow_matte)

        self.batch += 1

        if (self.running_std_mean == None):
            self.running_std_mean = torch.std_mean(shadow_matte)
        else:
            self.running_std_mean += torch.std_mean(shadow_matte)

        min = torch.min(shadow_matte)
        max = torch.max(shadow_matte)
        if(min < self.min):
            self.min = min

        if(max > self.max):
            self.max = max


    def get_std_mean(self):
        std_ave = self.running_std_mean[0] / self.batch
        mean_ave = self.running_std_mean[1] / self.batch
        return (std_ave, mean_ave)

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

class DatasetHSVAnalyzer():
    def __init__(self):
        self.running_std_mean_h = None
        self.running_std_mean_s = None
        self.running_std_mean_v = None
        self.batch = 0

        self.min = 100.0
        self.max = -100.0

    def compute_std_mean_of_batch(self, rgb_batch):
        rgb_batch = tensor_utils.normalize_to_01(rgb_batch)
        hsv_batch = kornia.color.rgb_to_hsv(rgb_batch)

        # print("HSV shape: ", np.shape(hsv_batch))
        self.batch += 1

        if (self.running_std_mean_h == None):
            self.running_std_mean_h = torch.std_mean(hsv_batch[:, 0])
        else:
            self.running_std_mean_h += torch.std_mean(hsv_batch[:, 0])

        if (self.running_std_mean_s == None):
            self.running_std_mean_s = torch.std_mean(hsv_batch[:, 1])
        else:
            self.running_std_mean_s += torch.std_mean(hsv_batch[:, 1])

        if (self.running_std_mean_v == None):
            self.running_std_mean_v = torch.std_mean(hsv_batch[:, 2])
        else:
            self.running_std_mean_v += torch.std_mean(hsv_batch[:, 2])

        min = torch.min(hsv_batch)
        max = torch.max(hsv_batch)
        if(min < self.min):
            self.min = min

        if(max > self.max):
            self.max = max

    def get_std_mean_h(self):
        std_ave = self.running_std_mean_h[0] / self.batch
        mean_ave = self.running_std_mean_h[1] / self.batch
        return (std_ave, mean_ave * 180.0)

    def get_std_mean_s(self):
        std_ave = self.running_std_mean_s[0] / self.batch
        mean_ave = self.running_std_mean_s[1] / self.batch
        return (std_ave * 100.0, mean_ave * 100.0)

    def get_std_mean_v(self):
        std_ave = self.running_std_mean_v[0] / self.batch
        mean_ave = self.running_std_mean_v[1] / self.batch
        return (std_ave * 100.0, mean_ave * 100.0)

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max



if __name__ == "__main__":
    main(sys.argv)