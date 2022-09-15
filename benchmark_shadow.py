import sys
from optparse import OptionParser

import torch
import cv2
import numpy as np
from torch import nn
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional
import constants
from loaders import dataset_loader

parser = OptionParser()
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/SID_PAMI/*.png")
parser.add_option('--ns_like_path', type=str, default = "E:/ISTD_Dataset/test/test_A/*.png")
parser.add_option('--ns_path', type=str, default = "E:/ISTD_Dataset/test/test_C/*.png")

class PAMIDataset(data.Dataset):
    def __init__(self, img_length, img_list_ns_like, img_list_gt):
        self.img_length = img_length
        self.img_list_a = img_list_ns_like
        self.img_list_b = img_list_gt

        self.transform_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]

        try:
            rgb_ns_like = cv2.imread(self.img_list_a[idx])
            rgb_ns_like = cv2.cvtColor(rgb_ns_like, cv2.COLOR_BGR2RGB)
            rgb_ns_like = self.transform_op(rgb_ns_like)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.transform_op(rgb_ns)


        except:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            rgb_ns_like = None
            rgb_ns = None

        return file_name, rgb_ns_like, rgb_ns

    def __len__(self):
        return self.img_length

def main(argv):
    (opts, args) = parser.parse_args(argv)

    ns_like_list = dataset_loader.assemble_img_list(opts.ns_like_path, opts)
    ns_list = dataset_loader.assemble_img_list(opts.ns_path, opts)

    img_length = len(ns_like_list)
    print("Length of images: %d %d" % (len(ns_like_list), len(ns_list)))

    data_loader = torch.utils.data.DataLoader(
        PAMIDataset(img_length, ns_like_list, ns_list),
        batch_size=256,
        num_workers=1,
        shuffle=False
    )

    mae = nn.L1Loss()
    mae_list = []
    for i, (_, rgb_ns_like, rgb_ns) in enumerate(data_loader, 0):
        mae_error = mae(rgb_ns_like, rgb_ns)
        print("Batch: ", i, " MAE error: ", mae_error)
        mae_list.append(mae_error)

    mean_mae = np.round(np.mean(mae_list) * 255.0, 4)
    print("Mean MAE: ", mean_mae)

if __name__ == "__main__":
    main(sys.argv)