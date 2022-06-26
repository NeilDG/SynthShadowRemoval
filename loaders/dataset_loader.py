import glob
import random
import torch
from torch.utils import data

import constants
from loaders import image_dataset
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

def load_map_train_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.MapDataset(a_list, path_c, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_map_test_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.MapDataset(a_list, path_c, 2, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
    )

    return data_loader

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

def load_map_train_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts):
    img_length = len(assemble_unpaired_data(rgb_dir, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDataset(img_length, rgb_dir, albedo_dir, shading_dir, shadow_dir, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_map_test_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts):
    img_length = len(assemble_unpaired_data(rgb_dir, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDataset(img_length, rgb_dir, albedo_dir, shading_dir, shadow_dir, 2, opts),
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_iid_datasetv2_train(rgb_dir_ws, rgb_dir_ns, albedo_dir, opts):
    rgb_list_ws = glob.glob(rgb_dir_ws)
    random.shuffle(rgb_list_ws)
    if (opts.img_to_load > 0):
        rgb_list_ws = rgb_list_ws[0: opts.img_to_load]

    for i in range(0, len(rgb_list_ws)):
        rgb_list_ws[i] = rgb_list_ws[i].replace("\\", "/")

    img_length = len(rgb_list_ws)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDatasetV2(img_length, rgb_list_ws, rgb_dir_ns, albedo_dir, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_iid_datasetv2_test(rgb_dir_ws, rgb_dir_ns, albedo_dir, opts):
    rgb_list_ws = glob.glob(rgb_dir_ws)
    random.shuffle(rgb_list_ws)
    if (opts.img_to_load > 0):
        rgb_list_ws = rgb_list_ws[0: opts.img_to_load]

    for i in range(0, len(rgb_list_ws)):
        rgb_list_ws[i] = rgb_list_ws[i].replace("\\", "/")

    img_length = len(rgb_list_ws)
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDatasetV2(img_length, rgb_list_ws, rgb_dir_ns, albedo_dir, 2, opts),
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

def load_shadowrelight_train_dataset(sample_path, path_rgb, path_a, path_b, opts):
    img_length = len(assemble_unpaired_data(sample_path, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadowRelightDatset(img_length, path_rgb, path_a, path_b, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shadowrelight_test_dataset(sample_path, path_rgb, path_a, path_b, opts):
    img_length = len(assemble_unpaired_data(sample_path, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadowRelightDatset(img_length, path_rgb, path_a, path_b, 2, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_shadowmap_train_recursive(path_a, folder_b, folder_c, return_shading: bool, opts):
    a_list = glob.glob(path_a + "/*/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDataset(a_list, folder_b, folder_c, 1, return_shading, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shadowmap_test_recursive(path_a, folder_b, folder_c, return_shading: bool, opts):
    a_list = glob.glob(path_a + "/*/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.IIDDataset(a_list, folder_b, folder_c, 2, return_shading, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
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
        batch_size=opts.img_per_iter,
        num_workers = opts.num_workers,
        shuffle=False,
        pin_memory=False

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

