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

    if(opts.map_choice == "smoothness"):
        data_loader = torch.utils.data.DataLoader(
            image_dataset.RenderSegmentDataset(a_list, path_c, 1),
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            shuffle=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            image_dataset.MapDataset(a_list, path_c, 1),
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            shuffle=True
        )

    return data_loader

def load_map_test_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load)
    print("Length of images: %d" % len(a_list))

    if(opts.map_choice == "smoothness"):
        data_loader = torch.utils.data.DataLoader(
            image_dataset.RenderSegmentDataset(a_list, path_c, 1),
            batch_size=4,
            num_workers=1,
            shuffle=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            image_dataset.MapDataset(a_list, path_c, 2),
            batch_size=4,
            num_workers=1,
            shuffle=True
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