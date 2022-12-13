import sys
from optparse import OptionParser
from pathlib import Path

import kornia.losses
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
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/AAAI_2020+_ISTD/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/AAAI_2020_ISTD/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/BMNET_2022_ISTD/ISTD_Result/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/SID_PAMI/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/DC-ShadowNet_ISTD/*.png")
parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/SynShadow-SP+M/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/SynShadow-DHAN/*.png")
# parser.add_option('--ns_like_path', type=str, default = "./comparison/ISTD Dataset/OURS/*.png")
# parser.add_option('--ns_like_path', type=str, default = "E:/ISTD_Dataset/test/test_A/*.png")
# parser.add_option('--ns_path', type=str, default = "E:/ISTD_Dataset/test/test_C/*.png")
# parser.add_option('--mask_path', type=str, default = "E:/ISTD_Dataset/test/test_B/*.png")

class ShadowDataset(data.Dataset):
    def __init__(self, img_length, img_list_ns_like, img_list_gt, img_list_mask):
        self.img_length = img_length
        self.img_list_a = img_list_ns_like
        self.img_list_b = img_list_gt
        self.img_list_mask = img_list_mask

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

            shadow_mask = cv2.imread(self.img_list_mask[idx])
            shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
            shadow_mask = self.transform_op(shadow_mask)


        except:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx], self.img_list_mask[idx])
            rgb_ns_like = None
            rgb_ns = None
            shadow_mask = None

        return file_name, rgb_ns_like, rgb_ns, shadow_mask

    def __len__(self):
        return self.img_length

class ShadowSRDDataset(data.Dataset):
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
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            rgb_ws = self.transform_op(rgb_ws)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.transform_op(rgb_ns)

            shadow_tensor = rgb_ns - rgb_ws
            shadow_matte = kornia.color.rgb_to_grayscale(shadow_tensor)
            shadow_mask = (shadow_matte >= 0.75).float()

        except:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            rgb_ws = None
            rgb_ns = None
            shadow_mask = None

        return file_name, rgb_ws, rgb_ns, shadow_mask

    def __len__(self):
        return self.img_length

def measure_performance(path_list, ns_path, mask_path, opts):
    for ns_like_path in path_list:
        model_name = ns_like_path.split("/")[-2]
        ns_like_list = dataset_loader.assemble_img_list(ns_like_path, opts)
        ns_list = dataset_loader.assemble_img_list(ns_path, opts)
        mask_list = dataset_loader.assemble_img_list(mask_path, opts)

        img_length = len(ns_like_list)
        print("%s: Length of images: %d %d" % (model_name, len(ns_like_list), len(ns_list)))

        data_loader = torch.utils.data.DataLoader(
            ShadowDataset(img_length, ns_like_list, ns_list, mask_list),
            batch_size=256,
            num_workers=1,
            shuffle=False
        )

        mae = nn.L1Loss()
        mse = nn.MSELoss()

        mae_list = []
        rmse_list = []
        psnr_list = []

        mae_list_ws = []
        rmse_list_ws = []
        psnr_list_ws = []

        mae_list_ns = []
        rmse_list_ns = []
        psnr_list_ns = []

        mae_lab = []
        mae_lab_ws = []
        mae_lab_ns = []

        rmse_lab = []
        rmse_lab_ws = []
        rmse_lab_ns = []

        for i, (_, rgb_ns_like, rgb_ns, shadow_mask) in enumerate(data_loader, 0):
            mae_error = mae(rgb_ns_like, rgb_ns)
            mae_list.append(mae_error)
            rmse_list.append(torch.sqrt(mse(rgb_ns_like, rgb_ns)))

            psnr_error = kornia.metrics.psnr(rgb_ns_like, rgb_ns, 1.0)
            psnr_list.append(psnr_error)

            rgb_ns_like_lab = kornia.color.rgb_to_lab(rgb_ns_like)
            rgb_ns_lab = kornia.color.rgb_to_lab(rgb_ns)

            mae_lab.append(mae(rgb_ns_like_lab, rgb_ns_lab))
            rmse_lab.append(torch.sqrt(mse(rgb_ns_like_lab, rgb_ns_lab)))

            #WS regions
            mae_list_ws.append(mae(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask))
            rmse_list_ws.append(torch.sqrt(mse(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask)))
            psnr_list_ws.append(kornia.metrics.psnr(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask, 1.0))
            mae_lab_ws.append(mae(rgb_ns_like_lab * shadow_mask, rgb_ns_lab * shadow_mask))
            rmse_lab_ws.append(torch.sqrt(mse(rgb_ns_like_lab * shadow_mask, rgb_ns_lab * shadow_mask)))

            #NS regions
            shadow_mask_inv = transforms.functional.invert(shadow_mask)
            mae_list_ns.append(mae(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv))
            psnr_list_ns.append(kornia.metrics.psnr(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv, 1.0))
            rmse_list_ns.append(torch.sqrt(mse(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv)))
            mae_lab_ns.append(mae(rgb_ns_like_lab * shadow_mask_inv, rgb_ns_lab * shadow_mask_inv))
            rmse_lab_ns.append(torch.sqrt(mse(rgb_ns_like_lab * shadow_mask_inv, rgb_ns_lab * shadow_mask_inv)))

        mean_mae = np.round(np.mean(mae_list) * 255.0, 4)
        mean_rmse = np.round(np.mean(rmse_list) * 255.0, 4)
        mean_psnr = np.round(np.mean(psnr_list), 4)
        mean_mae_lab = np.round(np.mean(mae_lab), 4)
        mean_rmse_lab = np.round(np.mean(rmse_lab), 4)

        mean_mae_ws = np.round(np.mean(mae_list_ws) * 255.0, 4)
        mean_rmse_ws = np.round(np.mean(rmse_list_ws) * 255.0, 4)
        mean_psnr_ws = np.round(np.mean(psnr_list_ws), 4)
        mean_mae_lab_ws = np.round(np.mean(mae_lab_ws), 4)
        mean_rmse_lab_ws = np.round(np.mean(rmse_lab_ws), 4)

        mean_mae_ns = np.round(np.mean(mae_list_ns) * 255.0, 4)
        mean_rmse_ns = np.round(np.mean(rmse_list_ns) * 255.0, 4)
        mean_psnr_ns = np.round(np.mean(psnr_list_ns), 4)
        mean_mae_lab_ns = np.round(np.mean(mae_lab_ns), 4)
        mean_rmse_lab_ns = np.round(np.mean(rmse_lab_ns), 4)

        # print("Model name: ", model_name, " Mean PSNR: ", mean_psnr, " Mean MAE: ", mean_mae, " Mean MAE Lab: ", mean_mae_lab,
        #       "\nModel name: ", model_name, " Mean PSNR (WS): ", mean_psnr_ws, " Mean MAE (WS): ", mean_mae_ws, " Mean MAE Lab (WS): ", mean_mae_lab_ws,
        #       "\nModel name: ", model_name, " Mean PSNR (NS): ", mean_psnr_ns, " Mean MAE (NS): ", mean_mae_ns, " Mean MAE Lab (NS): ", mean_mae_lab_ns)

        print("Model name: ", model_name, " Mean PSNR: ", mean_psnr, " Mean PSNR (WS): ", mean_psnr_ws, " Mean PSNR (NS): ", mean_psnr_ns,
              "\nModel name: ", model_name, " Mean RMSE: ", mean_rmse, " Mean RMSE Lab: ", mean_rmse_lab,
              "\nModel name: ", model_name, " Mean RMSE (WS): ", mean_rmse_ws, " Mean RMSE Lab (WS): ", mean_rmse_lab_ws,
              "\nModel name: ", model_name, " Mean RMSE (NS): ", mean_rmse_ns, " Mean RMSE Lab (NS): ", mean_rmse_lab_ns)

def measure_srd_performance(path_list, ns_path, opts):
    for ns_like_path in path_list:
        model_name = ns_like_path.split("/")[-2]
        ns_like_list = dataset_loader.assemble_img_list(ns_like_path, opts)
        ns_list = dataset_loader.assemble_img_list(ns_path, opts)

        img_length = len(ns_like_list)
        print("%s: Length of images: %d %d" % (model_name, len(ns_like_list), len(ns_list)))

        data_loader = torch.utils.data.DataLoader(
            ShadowSRDDataset(img_length, ns_like_list, ns_list),
            batch_size=256,
            num_workers=1,
            shuffle=False
        )

        mae = nn.L1Loss()
        mse = nn.MSELoss()

        mae_list = []
        rmse_list = []
        psnr_list = []

        mae_list_ws = []
        rmse_list_ws = []
        psnr_list_ws = []

        mae_list_ns = []
        rmse_list_ns = []
        psnr_list_ns = []

        mae_lab = []
        mae_lab_ws = []
        mae_lab_ns = []

        rmse_lab = []
        rmse_lab_ws = []
        rmse_lab_ns = []

        for i, (_, rgb_ns_like, rgb_ns, shadow_mask) in enumerate(data_loader, 0):
            mae_error = mae(rgb_ns_like, rgb_ns)
            mae_list.append(mae_error)
            rmse_list.append(torch.sqrt(mse(rgb_ns_like, rgb_ns)))

            psnr_error = kornia.metrics.psnr(rgb_ns_like, rgb_ns, 1.0)
            psnr_list.append(psnr_error)

            rgb_ns_like_lab = kornia.color.rgb_to_lab(rgb_ns_like)
            rgb_ns_lab = kornia.color.rgb_to_lab(rgb_ns)

            mae_lab.append(mae(rgb_ns_like_lab, rgb_ns_lab))
            rmse_lab.append(torch.sqrt(mse(rgb_ns_like_lab, rgb_ns_lab)))

            # WS regions
            mae_list_ws.append(mae(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask))
            rmse_list_ws.append(torch.sqrt(mse(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask)))
            psnr_list_ws.append(kornia.metrics.psnr(rgb_ns_like * shadow_mask, rgb_ns * shadow_mask, 1.0))
            mae_lab_ws.append(mae(rgb_ns_like_lab * shadow_mask, rgb_ns_lab * shadow_mask))
            rmse_lab_ws.append(torch.sqrt(mse(rgb_ns_like_lab * shadow_mask, rgb_ns_lab * shadow_mask)))

            # NS regions
            shadow_mask_inv = transforms.functional.invert(shadow_mask)
            mae_list_ns.append(mae(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv))
            psnr_list_ns.append(kornia.metrics.psnr(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv, 1.0))
            rmse_list_ns.append(torch.sqrt(mse(rgb_ns_like * shadow_mask_inv, rgb_ns * shadow_mask_inv)))
            mae_lab_ns.append(mae(rgb_ns_like_lab * shadow_mask_inv, rgb_ns_lab * shadow_mask_inv))
            rmse_lab_ns.append(torch.sqrt(mse(rgb_ns_like_lab * shadow_mask_inv, rgb_ns_lab * shadow_mask_inv)))

        mean_mae = np.round(np.mean(mae_list) * 255.0, 4)
        mean_rmse = np.round(np.mean(rmse_list) * 255.0, 4)
        mean_psnr = np.round(np.mean(psnr_list), 4)
        mean_mae_lab = np.round(np.mean(mae_lab), 4)
        mean_rmse_lab = np.round(np.mean(rmse_lab), 4)

        mean_mae_ws = np.round(np.mean(mae_list_ws) * 255.0, 4)
        mean_rmse_ws = np.round(np.mean(rmse_list_ws) * 255.0, 4)
        mean_psnr_ws = np.round(np.mean(psnr_list_ws), 4)
        mean_mae_lab_ws = np.round(np.mean(mae_lab_ws), 4)
        mean_rmse_lab_ws = np.round(np.mean(rmse_lab_ws), 4)

        mean_mae_ns = np.round(np.mean(mae_list_ns) * 255.0, 4)
        mean_rmse_ns = np.round(np.mean(rmse_list_ns) * 255.0, 4)
        mean_psnr_ns = np.round(np.mean(psnr_list_ns), 4)
        mean_mae_lab_ns = np.round(np.mean(mae_lab_ns), 4)
        mean_rmse_lab_ns = np.round(np.mean(rmse_lab_ns), 4)

        print("Model name: ", model_name, " Mean PSNR: ", mean_psnr, " Mean MAE: ", mean_mae, " Mean MAE Lab: ", mean_mae_lab,
              "\nModel name: ", model_name, " Mean PSNR (WS): ", mean_psnr_ws, " Mean MAE (WS): ", mean_mae_ws, " Mean MAE Lab (WS): ", mean_mae_lab_ws,
              "\nModel name: ", model_name, " Mean PSNR (NS): ", mean_psnr_ns, " Mean MAE (NS): ", mean_mae_ns, " Mean MAE Lab (NS): ", mean_mae_lab_ns)

        print("Model name: ", model_name, " Mean RMSE: ", mean_rmse, " Mean RMSE Lab: ", mean_rmse_lab,
              "\nModel name: ", model_name, " Mean RMSE (WS): ", mean_rmse_ws, " Mean RMSE Lab (WS): ", mean_rmse_lab_ws,
              "\nModel name: ", model_name, " Mean RMSE (NS): ", mean_rmse_ns, " Mean RMSE Lab (NS): ", mean_rmse_lab_ns)

def save_img_copies_for_results(results_list, ns_path, dataset_name, target_size, opts):
    transform_op = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()])

    BASE_SAVE_PATH = "./reports/"

    results_list.append(ns_path)
    for img_dir in results_list:
        img_list = dataset_loader.assemble_img_list(img_dir, opts)
        img_dir_split = img_dir.split("/")

        folder_dir = BASE_SAVE_PATH + dataset_name + "/" + img_dir_split[3] + "/"
        try:
            path = Path(folder_dir)
            path.mkdir(parents=True)
        except OSError as error:
            print("Save path already exists. Skipping.", error)

        for img_path in img_list:
            file_name = img_path.split("/")[4]
            rgb_img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = transform_op(rgb_img)

            print(folder_dir + file_name)
            torchvision.utils.save_image(rgb_img, folder_dir + file_name)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    istd_all_list = [
    # "E:/ISTD_Dataset/test/test_A/*.png",
    # "./comparison/ISTD Dataset/SID_PAMI/*.png",
    # "./comparison/ISTD Dataset/DC-ShadowNet_ISTD/*.png",
    # "./comparison/ISTD Dataset/BMNET_2022_ISTD/*.png",
    # "./comparison/ISTD Dataset/AAAI_2020_ISTD/*.png",
    # "./comparison/ISTD Dataset/AAAI_2020+_ISTD/*.png",
    # "./comparison/ISTD Dataset/SynShadow-SP+M/*.png",
    # "./comparison/ISTD Dataset/SynShadow-DHAN/*.png",
    "./comparison/ISTD Dataset/OURS/*.png"]

    ns_path = "E:/ISTD_Dataset/test/test_C/*.png"
    mask_path = "E:/ISTD_Dataset/test/test_B/*.png"

    # measure_performance(istd_all_list, ns_path, mask_path, opts)
    # save_img_copies_for_results(istd_all_list, ns_path, "ISTD Dataset", (240, 320), opts)

    # for SRD
    ns_path = "E:/SRD_Test/srd/shadow_free/*.jpg"
    mask_path = "E:/SRD_Test/srd/mask/*.jpg"

    sbu_all_list = [
    # "E:/SRD_Test/srd/shadow/*.jpg",
    # "./comparison/SRD Dataset/SID_PAMI/*.png",
    # "./comparison/SRD Dataset/DC-ShadowNet/*.png",
    # "./comparison/SRD Dataset/BMNET_2022/*.jpg",
    # "./comparison/SRD Dataset/AAAI_2020_SRD/*.jpg",
    # "./comparison/SRD Dataset/AAAI_2020+_SRD/*.jpg",
    # "./comparison/SRD Dataset/SynShadow-SP+M/*.png",
    # "./comparison/SRD Dataset/SynShadow-DHAN/*.png",
    "./comparison/SRD Dataset/OURS/*.png"]

    measure_performance(sbu_all_list, ns_path, mask_path, opts)
    save_img_copies_for_results(sbu_all_list, ns_path, "SRD Dataset", (160, 210), opts)



if __name__ == "__main__":
    main(sys.argv)