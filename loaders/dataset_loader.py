import glob
import random
import torch
from torch.utils import data

import constants
from config import iid_server_config
from loaders import image_dataset, iid_test_datasets
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

def assemble_img_list(img_dir, opts, shuffle = False):
    img_list = glob.glob(img_dir)
    if(shuffle):
        random.shuffle(img_list)

    if (opts.img_to_load > 0):
        img_list = img_list[0: opts.img_to_load]

    for i in range(0, len(img_list)):
        img_list[i] = img_list[i].replace("\\", "/")

    return img_list

def load_relighting_train_dataset(rgb_dir, albedo_dir, scene_root, opts):
    albedo_list = glob.glob(albedo_dir + "/*.png")
    scene_list = os.listdir(scene_root)

    print("Image length: %d Number of known scenes: %d" % (len(albedo_list), len(scene_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RelightDataset(len(albedo_list), rgb_dir, albedo_dir, scene_list, opts),
        batch_size=opts.batch_size,
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

def load_cgi_dataset(rgb_dir, patch_size, opts):
    rgb_list = assemble_img_list(rgb_dir, opts)

    img_length = len(rgb_list)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        iid_test_datasets.CGIDataset(img_length, rgb_list, 2, patch_size),
        batch_size=8,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_iiw_dataset(rgb_dir, opts):
    rgb_list = assemble_img_list(rgb_dir, opts)
    rgb_list = rgb_list[0:100] #temporary short

    img_length = len(rgb_list)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        iid_test_datasets.IIWDataset(img_length, rgb_list),
        batch_size=8,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_bell2014_dataset(r_dir, s_dir, patch_size, opts):
    r_list = assemble_img_list(r_dir, opts)
    s_list = assemble_img_list(s_dir, opts)

    img_length = len(r_list)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        iid_test_datasets.Bell2014Dataset(img_length, r_list, s_list, 2, patch_size),
        batch_size=8,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_shadow_train_dataset(ws_path, ns_path, ws_istd, ns_istd, patch_size, batch_size,
                              mix_type, opts):
    ws_list = assemble_img_list(ws_path, opts)
    ns_list = assemble_img_list(ns_path, opts)

    ws_istd_list = assemble_img_list(ws_istd, opts)
    ns_istd_list = assemble_img_list(ns_istd, opts)

    if(mix_type == 1):
        print("Mixing ISTD and synthetic train datasets")
        for i in range(0, 1):
            ws_list += ws_list
            ns_list += ns_list

        for i in range(0, 5):
            ws_list += ws_istd_list
            ns_list += ns_istd_list
    elif(mix_type == 2):
        print("Using ISTD dataset only")
        # to be efficient in batch size + data loading, duplicate list
        ws_list = ws_istd_list
        ns_list = ns_istd_list
        for i in range(0, 5):
            ws_list += ws_istd_list
            ns_list += ns_istd_list

    else:
        print("Using synthetic train dataset")
        for i in range(0, 1):
            ws_list += ws_list
            ns_list += ns_list

    img_length = len(ws_list)
    print("Length of images: %d %d" % (len(ws_list), len(ns_list)))

    data_loader = torch.utils.data.DataLoader(
        iid_test_datasets.ShadowTrainDataset(img_length, ws_list, ns_list, 1, patch_size),
        batch_size=batch_size,
        num_workers=opts.num_workers,
        shuffle=False
    )

    return data_loader

def load_shadow_test_dataset(ws_path, ns_path, opts):
    ws_list = assemble_img_list(ws_path, opts)
    ns_list = assemble_img_list(ns_path, opts)

    img_length = len(ws_list)
    print("Length of images: %d %d" % (len(ws_list), len(ns_list)))

    data_loader = torch.utils.data.DataLoader(
        iid_test_datasets.ShadowTestDataset(img_length, ws_list, ns_list, 2),
        batch_size=16,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_iid_datasetv2_test(rgb_dir_ws, rgb_dir_ns, unlit_dir, albedo_dir, patch_size, opts):
    rgb_list_ws = glob.glob(rgb_dir_ws)
    random.shuffle(rgb_list_ws)
    if (opts.img_to_load > 0):
        rgb_list_ws = rgb_list_ws[0: opts.img_to_load]

    for i in range(0, len(rgb_list_ws)):
        rgb_list_ws[i] = rgb_list_ws[i].replace("\\", "/")

    img_length = len(rgb_list_ws)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDatasetV2(img_length, rgb_list_ws, rgb_dir_ns, unlit_dir, albedo_dir, 2, patch_size),
        batch_size=4,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_single_test_dataset(path_a):
    a_list = glob.glob(path_a)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RealWorldDataset(a_list),
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_gta_dataset(rgb_dir, albedo_dir, opts):
    rgb_list = assemble_unpaired_data(rgb_dir, opts.img_to_load)
    albedo_list = assemble_unpaired_data(albedo_dir, opts.img_to_load)
    print("Length of images: %d %d" % (len(rgb_list), len(albedo_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.GTATestDataset(rgb_list, albedo_list, opts),
        batch_size=4,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )

    return data_loader

def load_da_dataset_train(imgx_dir, imgy_dir, opts):
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    network_config = sc_instance.interpret_style_transfer_config_from_version()

    imgx_list = glob.glob(imgx_dir)
    imgy_list = glob.glob(imgy_dir)

    random.shuffle(imgx_list)
    random.shuffle(imgy_list)

    if(opts.img_to_load > 0):
        imgx_list = imgx_list[0: opts.img_to_load]
        imgy_list = imgy_list[0: opts.img_to_load]

    print("Length of images: %d %d" % (len(imgx_list), len(imgy_list)))

    if(len(imgx_list) == 0 or len(imgy_list) == 0):
        return None

    data_loader = torch.utils.data.DataLoader(
        image_dataset.GenericPairedDataset(imgx_list, imgy_list, 1, network_config["patch_size"]),
        batch_size=network_config["img_per_iter"],
        num_workers = opts.num_workers,
        shuffle=False,
        pin_memory=True

    )

    return data_loader

def load_unlit_dataset_train(styled_dir, unlit_dir, opts):
    styled_img_list = glob.glob(styled_dir)
    random.shuffle(styled_img_list)

    if (opts.img_to_load > 0):
        styled_img_list = styled_img_list[0: opts.img_to_load]

    print("Length of images: %d " % len(styled_img_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.UnlitDataset(len(styled_img_list), styled_img_list, unlit_dir, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=False
    )

    return data_loader

def load_unlit_dataset_test(styled_dir, unlit_dir, opts):
    styled_img_list = glob.glob(styled_dir)
    random.shuffle(styled_img_list)

    if (opts.img_to_load > 0):
        styled_img_list = styled_img_list[0: opts.img_to_load]

    print("Length of images: %d " % len(styled_img_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.UnlitDataset(len(styled_img_list), styled_img_list, unlit_dir, 2, opts),
        batch_size=4,
        num_workers=1,
        shuffle=False
    )

    return data_loader


def load_da_dataset_test(imgx_dir, imgy_dir, opts):
    sc_instance = iid_server_config.IIDServerConfig.getInstance()
    network_config = sc_instance.interpret_style_transfer_config_from_version()

    imgx_list = glob.glob(imgx_dir)
    imgy_list = glob.glob(imgy_dir)

    random.shuffle(imgx_list)
    random.shuffle(imgy_list)

    if (opts.img_to_load > 0):
        imgx_list = imgx_list[0: opts.img_to_load]
        imgy_list = imgy_list[0: opts.img_to_load]

    print("Length of images: %d %d" % (len(imgx_list), len(imgy_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.GenericPairedDataset(imgx_list, imgy_list, 2, network_config["patch_size"]),
        batch_size=4,
        num_workers=1,
        shuffle=False,
        pin_memory=False
    )

    return data_loader

def load_ffa_dataset_train(imgx_dir, imgy_dir, opts):
    imgx_list = glob.glob(imgx_dir)
    imgy_list = glob.glob(imgy_dir)

    random.shuffle(imgx_list)
    random.shuffle(imgy_list)

    if(opts.img_to_load > 0):
        imgx_list = imgx_list[0: opts.img_to_load]
        imgy_list = imgy_list[0: opts.img_to_load]

    print("Length of images: %d %d" % (len(imgx_list), len(imgy_list)))

    if(len(imgx_list) == 0 or len(imgy_list) == 0):
        return None

    data_loader = torch.utils.data.DataLoader(
        image_dataset.GenericPairedDataset(imgx_list, imgy_list, 1, opts),
        batch_size=opts.batch_size,
        num_workers = opts.num_workers,
        shuffle=False,
        pin_memory=False

    )

    return data_loader


def load_ffa_dataset_test(imgx_dir, imgy_dir, opts):
    imgx_list = glob.glob(imgx_dir)
    imgy_list = glob.glob(imgy_dir)

    random.shuffle(imgx_list)
    random.shuffle(imgy_list)

    if (opts.img_to_load > 0):
        imgx_list = imgx_list[0: opts.img_to_load]
        imgy_list = imgy_list[0: opts.img_to_load]

    print("Length of images: %d %d" % (len(imgx_list), len(imgy_list)))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.GenericPairedDataset(imgx_list, imgy_list, 2, opts),
        batch_size=4,
        num_workers=1,
        shuffle=False,
        pin_memory=False
    )

    return data_loader

