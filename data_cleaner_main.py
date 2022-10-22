import sys
from optparse import OptionParser

from tqdm import tqdm

import constants
from config import iid_server_config
from loaders import dataset_loader
import torch

from utils import tensor_utils

parser = OptionParser()
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--version', type=str, default="v58.16")
parser.add_option('--dataset_version_to_refine', type=str, default="v26")
parser.add_option('--iteration', type=int, help="Style version?", default="1")
parser.add_option('--num_workers', type=int, help="Workers", default="12")

# Running mean of Synth dataset is:  tensor(7.9453e-05)  Std dev:  tensor(0.0002)
# Synth dataset min:  tensor(-0.0189) tensor(0.8824)

def quantify_datasets(opts):
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    network_config = sc_instance.interpret_shadow_matte_params_from_version()
    dataset_version = network_config["dataset_version"]

    constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=dataset_version)

    ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "E:/ISTD_Dataset/test/test_B/*.png"

    load_size = 256
    istd_loader, dataset_count = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, 256, opts)

    needed_progress = int(dataset_count / load_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    istd_analyzer = DatasetAnalyzer()
    for i, (file_name, _, _, shadow_matte) in enumerate(istd_loader, 0):
        istd_analyzer.compute_std_mean_of_batch(shadow_matte)
        pbar.update(1)

    pbar.close()

    train_loader, dataset_count = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, 256, 256, opts)

    needed_progress = int(dataset_count / load_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    synth_analyzer = DatasetAnalyzer()
    for i, (file_name, _, _, _, shadow_matte) in enumerate(train_loader, 0):
        synth_analyzer.compute_std_mean_of_batch(shadow_matte)
        pbar.update(1)

    pbar.close()

    istd_mean = istd_analyzer.get_std_mean()[1]
    istd_std = istd_analyzer.get_std_mean()[0]
    print("Running mean of ISTD is: ", istd_mean, " Std dev: ", istd_std)
    print("ISTD min: ", istd_analyzer.get_min(), istd_analyzer.get_max())

    print("Running mean of Synth dataset is: ", synth_analyzer.get_std_mean()[1], " Std dev: ", synth_analyzer.get_std_mean()[0])
    print("Synth dataset min: ", synth_analyzer.get_min(), synth_analyzer.get_max())

def prepare_clean(opts):
    constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"
    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=opts.dataset_version_to_refine)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=opts.dataset_version_to_refine)

    ws_istd = "E:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "E:/ISTD_Dataset/test/test_C/*.png"

    ws_list = dataset_loader.assemble_img_list(rgb_dir_ws, opts)
    ns_list = dataset_loader.assemble_img_list(rgb_dir_ns, opts)

    istd_mean = -0.0003
    istd_std = 0.0345 * 2.0

    print("Dataset len before: ", len(ws_list))
    index = 199262
    ws_list = ws_list[index: len(ws_list)]
    ns_list = ns_list[index: len(ns_list)]

    print("Trimming dataset. Dataset len after: ", len(ws_list))

    dataset_loader.clean_dataset_using_std_mean(ws_list, ns_list, istd_mean, istd_std)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    constants.shadow_network_version = opts.version
    constants.shadow_matte_network_version = opts.version
    iid_server_config.IIDServerConfig.initialize()

    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

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



if __name__ == "__main__":
    main(sys.argv)