import glob
import random
import torch
from torch.utils import data
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

def load_map_train_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts):
    img_length = len(assemble_unpaired_data(albedo_dir, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ImageRelightDataset(img_length, rgb_dir, albedo_dir, shading_dir, shadow_dir, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_map_test_recursive(rgb_dir, albedo_dir, shading_dir, shadow_dir, opts):
    img_length = len(assemble_unpaired_data(albedo_dir, opts.img_to_load))
    print("Length of images: %d" % img_length)

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ImageRelightDataset(img_length, rgb_dir, albedo_dir, shading_dir, shadow_dir, 2, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_map_test_recursive_2(path_a, path_c, opts):
    a_list = glob.glob(path_a + "/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.MapDataset(a_list, path_c, 2, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_color_train_dataset(path_a, path_c, path_segment, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load / 2)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, path_c, path_segment, 1),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_color_test_dataset(path_a, path_c, path_segment, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load / 2)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, path_c, path_segment, 2),
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_single_train_dataset(path_a, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)

    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RealWorldTrainDataset(a_list),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_single_test_dataset(path_a, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RealWorldDataset(a_list),
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_render_train_dataset(path_a, path_b, path_c, path_d, path_e, path_f, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RenderDataset(a_list, path_b, path_c, path_d, path_e, path_f, 1),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_render_test_dataset(path_a, path_b, path_c, path_d, path_e, path_f, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.RenderDataset(a_list, path_b, path_c, path_d, path_e, path_f, 2),
        batch_size=4,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_shading_train_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadingDataset(a_list, path_c, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shading_test_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadingDataset(a_list, path_c, 2, opts),
        batch_size=16,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_shading_train_recursive(path_a, path_c, opts):
    a_list = glob.glob(path_a + "/*/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadingDataset(a_list, path_c, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shading_test_recursive(path_a, path_c, opts):
    a_list = glob.glob(path_a + "/*/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadingDataset(a_list, path_c, 2, opts),
        batch_size=16,
        num_workers=1,
        shuffle=False
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
        image_dataset.ImageRelightDataset(a_list, folder_b, folder_c, 1, return_shading, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shadowmap_test_recursive(path_a, folder_b, folder_c, return_shading: bool, opts):
    a_list = glob.glob(path_a + "/*/rgb/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ImageRelightDataset(a_list, folder_b, folder_c, 2, return_shading, opts),
        batch_size=2,
        num_workers=1,
        shuffle=False
    )

    return data_loader

def load_shadow_priors_train(path_a, opts):
    a_list = glob.glob(path_a + "/*/shadow_map/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadowPriorDataset(a_list, 1, opts),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_shadow_priors_test(path_a, opts):
    a_list = glob.glob(path_a + "/*/shadow_map/*.png")
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ShadowPriorDataset(a_list, 1, opts),
        batch_size=2,
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
        shuffle=False
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
        batch_size=opts.batch_size,
        num_workers = opts.num_workers,
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
        shuffle=False
    )

    return data_loader

