import glob
import random
from pathlib import Path

import kornia
import numpy as np
import torch
import torchvision.utils
from torch.utils import data
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm

import global_config
from config.network_config import ConfigHolder
from loaders import shadow_datasets, image_datasets
import os

def assemble_unpaired_data(path_a, num_image_to_load=-1, force_complete=False):
    a_list = []

    loaded = 0
    for (root, dirs, files) in os.walk(path_a):
        for f in files:
            file_name = os.path.join(root, f)
            a_list.append(file_name)
            loaded = loaded + 1
            if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                break

    while loaded != num_image_to_load and force_complete:
        for (root, dirs, files) in os.walk(path_a):
            for f in files:
                file_name = os.path.join(root, f)
                a_list.append(file_name)
                loaded = loaded + 1
                if (num_image_to_load != -1 and len(a_list) == num_image_to_load):
                    break

    return a_list

def assemble_img_list(img_dir):
    img_list = glob.glob(img_dir)

    if (global_config.img_to_load > 0):
        img_list = img_list[0: global_config.img_to_load]

    for i in range(0, len(img_list)):
        img_list[i] = img_list[i].replace("\\", "/")

    return img_list

def load_relighting_train_dataset(rgb_dir, albedo_dir, scene_root, opts):
    albedo_list = glob.glob(albedo_dir + "/*.png")
    scene_list = os.listdir(scene_root)

    print("Image length: %d Number of known scenes: %d" % (len(albedo_list), len(scene_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RelightDataset(len(albedo_list), rgb_dir, albedo_dir, scene_list, opts),
        batch_size=opts.load_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_relighting_test_dataset(rgb_dir, albedo_dir, scene_root, opts):
    albedo_list = glob.glob(albedo_dir + "/*.png")
    scene_list = os.listdir(scene_root)

    print("Image length: %d Number of known scenes: %d" % (len(albedo_list), len(scene_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RelightDataset(len(albedo_list), rgb_dir, albedo_dir, scene_list, opts),
        batch_size=16,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_iid_datasetv2_train(rgb_dir_ws, rgb_dir_ns, unlit_dir, albedo_dir, patch_size, batch_size, opts):
    rgb_list_ws = glob.glob(rgb_dir_ws)
    random.shuffle(rgb_list_ws)
    if (opts.img_to_load > 0):
        rgb_list_ws = rgb_list_ws[0: opts.img_to_load]

    for i in range(0, len(rgb_list_ws)):
        rgb_list_ws[i] = rgb_list_ws[i].replace("\\", "/")

    img_length = len(rgb_list_ws)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDatasetV2(img_length, rgb_list_ws, rgb_dir_ns, unlit_dir, albedo_dir, 1, patch_size),
        batch_size=batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def clean_dataset(ws_path, ns_path, filter_minimum):
    BASE_PATH = "E:/SynthWeather Dataset 10/"
    ws_path_revised = ws_path[0].split("/")
    ns_path_revised = ns_path[0].split("/")

    SAVE_PATH_WS = BASE_PATH + ws_path_revised[2] + "_refined/" + ws_path_revised[3] + "/" + ws_path_revised[4] + "/"
    SAVE_PATH_NS = BASE_PATH + ns_path_revised[2] + "_refined/" + ns_path_revised[3] + "/" + ns_path_revised[4] + "/"

    try:
        path = Path(SAVE_PATH_WS)
        path.mkdir(parents=True)

        path = Path(SAVE_PATH_NS)
        path.mkdir(parents=True)
    except OSError as error:
        print("Save path already exists. Skipping.", error)

    assert filter_minimum <= 100.0, "Filter minimum cannot be > 100.0"

    index = 0
    for (ws_img_path, ns_img_path) in zip(ws_path, ns_path):
        ws_img = cv2.imread(ws_img_path)
        # ws_img = cv2.cvtColor(ws_img, cv2.COLOR_BGR2RGB)

        sobel_x = cv2.Sobel(ws_img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(ws_img, cv2.CV_64F, 0, 1, ksize=5)
        sobel_img = sobel_x + sobel_y
        total_pixels = np.shape(ws_img)[0] * np.shape(ws_img)[1]
        sobel_quality = np.linalg.norm(sobel_img) / total_pixels

        if(sobel_quality > filter_minimum): #only save images with good edges
            img_name = ws_img_path.split(".")[0].split("/")[-1]
            ns_img = cv2.imread(ns_img_path)
            # ns_img = cv2.cvtColor(ns_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(SAVE_PATH_WS + img_name + ".png", ws_img)
            cv2.imwrite(SAVE_PATH_NS + img_name + ".png", ns_img)
            print("Saved image: ", (SAVE_PATH_WS + img_name + ".png"))
        else:
            print("Sobel quality of img %s: %f. DISCARDING" % (ws_img_path, sobel_quality))

def clean_dataset_using_std_mean(ws_path, ns_path, basis_mean, basis_std):
    BASE_PATH = "E:/SynthWeather Dataset 10/"
    ws_path_revised = ws_path[0].split("/")
    ns_path_revised = ns_path[0].split("/")

    print(ws_path_revised)

    SAVE_PATH_WS = BASE_PATH + ws_path_revised[2] + "_refined/" + ws_path_revised[3] + "/" + ws_path_revised[4] + "/"
    SAVE_PATH_NS = BASE_PATH + ns_path_revised[2] + "_refined/" + ns_path_revised[3] + "/" + ns_path_revised[4] + "/"

    try:
        path = Path(SAVE_PATH_WS)
        path.mkdir(parents=True)

        path = Path(SAVE_PATH_NS)
        path.mkdir(parents=True)
    except OSError as error:
        print("Save path already exists. Skipping.", error)

    assert basis_std <= 100.0, "Filter minimum cannot be > 100.0"

    tensor_op = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(global_config.TEST_IMAGE_SIZE),
        transforms.ToTensor()])

    num_saved = 0
    num_discarded = 0

    needed_progress = len(ws_path)
    pbar = tqdm(total=needed_progress)
    for (ws_img_path, ns_img_path) in zip(ws_path, ns_path):
        rgb_ws = cv2.imread(ws_img_path)
        rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
        rgb_ws = tensor_op(rgb_ws)

        rgb_ns = cv2.imread(ns_img_path)
        rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
        rgb_ns = tensor_op(rgb_ns)

        shadow_map = rgb_ns - rgb_ws
        shadow_matte = kornia.color.rgb_to_grayscale(shadow_map)

        sm_mean = torch.mean(shadow_matte)

        min = basis_mean - basis_std
        max = basis_mean + basis_std

        if(min <= sm_mean <= max): #only save images within the specified mean and std
            img_name = ws_img_path.split(".")[0].split("/")[-1]

            torchvision.utils.save_image(rgb_ws, SAVE_PATH_WS + img_name + ".png")
            torchvision.utils.save_image(rgb_ns, SAVE_PATH_NS + img_name + ".png")
            # print("Saved image: ", (SAVE_PATH_WS + img_name + ".png"))

            num_saved += 1
        else:
            # print("Mean quality of img %s: %f MIN: %f MAX: %f. DISCARDING" % (ws_img_path, sm_mean, min, max))
            num_discarded +=1

        pbar.update(1)

    pbar.close()
    print("Total images processed: " + str(len(ws_path)) + " Saved: " + str(num_saved) + " Discarded: " + str(num_discarded))


def load_shadow_train_dataset():
    initial_ws_list = glob.glob(global_config.rgb_dir_ws)
    initial_ns_list = glob.glob(global_config.rgb_dir_ns)

    if(global_config.img_to_load > 0):
        initial_ws_list = initial_ws_list[:global_config.img_to_load]
        initial_ns_list = initial_ns_list[:global_config.img_to_load]

    # print("Length: ", len(initial_ws_list), len(initial_ns_list))

    temp_list = list(zip(initial_ws_list, initial_ns_list))
    random.shuffle(temp_list)
    initial_ws_list, initial_ns_list = zip(*temp_list)

    initial_istd_ws_list = glob.glob(global_config.ws_istd)
    initial_istd_ns_list = glob.glob(global_config.ns_istd)

    temp_list = list(zip(initial_istd_ws_list, initial_istd_ns_list))
    random.shuffle(temp_list)
    initial_istd_ws_list, initial_istd_ns_list = zip(*temp_list)

    initial_srd_ws_list = glob.glob(global_config.ws_srd)
    initial_srd_ns_list = glob.glob(global_config.ns_srd)

    temp_list = list(zip(initial_srd_ws_list, initial_srd_ns_list))
    random.shuffle(temp_list)
    initial_srd_ws_list, initial_srd_ns_list = zip(*temp_list)

    ws_list = []
    ns_list = []

    network_config = ConfigHolder.getInstance().get_network_config()
    for i in range(0, network_config["dataset_repeats"]): #TEMP: formerly 0-1
        ws_list += initial_ws_list
        ns_list += initial_ns_list

    mix_istd = ConfigHolder.getInstance().get_network_attribute("mix_istd", 0.0)
    if(mix_istd > 0.0):
        synth_len = int(len(ws_list) * network_config["mix_istd"]) #add N% istd
        istd_len = 0
        while istd_len < synth_len:
            ws_list += initial_istd_ws_list
            ns_list += initial_istd_ns_list
            istd_len += len(initial_istd_ws_list)
    else:
        istd_len = 0

    mix_srd = ConfigHolder.getInstance().get_network_attribute("mix_srd", 0.0)
    if (mix_srd > 0.0):
        synth_len = int(len(ws_list) * network_config["mix_srd"])  # add N% istd
        srd_len = 0
        while srd_len < synth_len:
            ws_list += initial_srd_ws_list
            ns_list += initial_srd_ns_list
            srd_len += len(initial_srd_ws_list)
    else:
        srd_len = 0

    img_length = len(ws_list)
    print("Length of images: %d %d. ISTD len: %d. SRD len: %d"  % (len(ws_list), len(ns_list), istd_len, srd_len))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowTrainDataset(img_length, ws_list, ns_list, 1),
        batch_size=global_config.load_size,
        num_workers=int(global_config.num_workers / 2),
        shuffle=False,
    )

    return data_loader, len(ws_list)

def load_shadow_test_dataset():
    ws_list = glob.glob(global_config.rgb_dir_ws)
    ns_list = glob.glob(global_config.rgb_dir_ns)

    if (global_config.img_to_load > 0):
        ws_list = ws_list[:global_config.img_to_load]
        ns_list = ns_list[:global_config.img_to_load]

    img_length = len(ws_list)
    print("Length of images: %d %d" % (len(ws_list), len(ns_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowTrainDataset(img_length, ws_list, ns_list, 2),
        batch_size=global_config.test_size,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_list)

def load_shadow_matte_dataset(ws_path, sm_path):
    ws_list = glob.glob(ws_path)
    sm_list = glob.glob(sm_path)

    img_length = len(ws_list)
    print("Length of images: %d %d" % (len(ws_list), len(sm_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowTrainDataset(img_length, ws_list, sm_list, 2),
        batch_size=8,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_list)

def load_istd_train_dataset():
    ws_istd_list = glob.glob(global_config.ws_istd)
    ns_istd_list = glob.glob(global_config.ns_istd)

    img_length = len(ws_istd_list)
    print("Length of images: %d %d" % (len(ws_istd_list), len(ns_istd_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowTrainDataset(img_length, ws_istd_list, ns_istd_list, 1),
        batch_size=global_config.load_size,
        num_workers=4,
        shuffle=False
    )

    return data_loader, len(ws_istd_list)

def load_istd_dataset(with_shadow_mask = False):
    ws_istd_list = glob.glob(global_config.ws_istd)
    ns_istd_list = glob.glob(global_config.ns_istd)
    mask_istd_list = glob.glob(global_config.mask_istd)

    img_length = len(ws_istd_list)
    print("Length of images: %d %d %d" % (len(ws_istd_list), len(ns_istd_list), len(mask_istd_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowISTDDataset(img_length, ws_istd_list, ns_istd_list, mask_istd_list, 1, with_shadow_mask),
        batch_size=global_config.test_size,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_istd_list)

def load_srd_dataset(with_shadow_mask = False):
    ws_list = glob.glob(global_config.ws_srd)
    ns_list = glob.glob(global_config.ns_srd)
    mask_list = glob.glob(global_config.mask_srd)

    img_length = len(ws_list)
    print("Length of images: %d %d %d" % (len(ws_list), len(ns_list), len(mask_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowSRDDataset(img_length, ws_list, ns_list, mask_list, 1, with_shadow_mask),
        batch_size=global_config.test_size,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_list)

def load_istd_srd_dataset(with_shadow_mask = False):
    ws_istd_list = glob.glob(global_config.ws_istd)
    ns_istd_list = glob.glob(global_config.ns_istd)
    mask_istd_list = glob.glob(global_config.mask_istd)

    ws_list = glob.glob(global_config.ws_srd)
    ns_list = glob.glob(global_config.ns_srd)
    mask_list = glob.glob(global_config.mask_srd)

    ws_list = ws_istd_list + ws_list
    ns_list = ns_istd_list + ns_list
    mask_list = mask_istd_list + mask_list

    #Mix together
    temp_list = list(zip(ws_list, ns_list, mask_list))
    random.shuffle(temp_list)
    ws_list, ns_list, mask_list = zip(*temp_list)

    img_length = len(ws_list)
    print("Length of images: %d %d %d" % (len(ws_list), len(ns_list), len(mask_list)))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowISTDDataset(img_length, ws_list, ns_list, mask_list, 1, with_shadow_mask),
        batch_size=global_config.test_size,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_list)


def load_usr_dataset():
    ws_list = glob.glob(global_config.usr_test)

    img_length = len(ws_list)
    print("Length of images: %d" % len(ws_list))

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowUSRTestDataset(img_length, ws_list, 1),
        batch_size=global_config.test_size,
        num_workers=1,
        shuffle=False
    )

    return data_loader, len(ws_list)

def load_single_test_dataset(path_a, opts):
    print("Dataset path: ", path_a)
    a_list = glob.glob(path_a)
    random.shuffle(a_list)
    if (opts.img_to_load > 0):
        a_list = a_list[0: opts.img_to_load]

    # a_list = a_list[100000:328497] #TODO: Temp only

    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 2),
        batch_size=128,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_train_img2img_dataset(a_path, b_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)
    a_list_dup = glob.glob(a_path)
    b_list_dup = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]
        a_list_dup = a_list_dup[0: global_config.img_to_load]
        b_list_dup = b_list_dup[0: global_config.img_to_load]

    for i in range(0, network_config["dataset_a_repeats"]): #TEMP: formerly 0-1
        a_list += a_list_dup

    for i in range(0, network_config["dataset_b_repeats"]): #TEMP: formerly 0-1
        b_list += b_list_dup

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d"  % (img_length, len(b_list)))

    num_workers = global_config.num_workers
    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers
    )

    return data_loader, img_length

def load_test_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    random.shuffle(a_list)
    random.shuffle(b_list)

    img_length = len(a_list)
    print("Length of images: %d %d" % (img_length, len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.test_size,
        num_workers=1
    )

    return data_loader, img_length

def load_singleimg_dataset(a_path):
    a_list = glob.glob(a_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]

    random.shuffle(a_list)

    img_length = len(a_list)
    print("Length of images: %d" % (img_length))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 1),
        batch_size=global_config.test_size,
        num_workers=4
    )

    return data_loader, img_length

