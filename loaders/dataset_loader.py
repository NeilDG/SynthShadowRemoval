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

def load_color_train_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load / 2)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, path_c, 1),
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=True
    )

    return data_loader

def load_color_test_dataset(path_a, path_c, opts):
    a_list = assemble_unpaired_data(path_a, opts.img_to_load / 2)
    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_dataset.ColorTransferDataset(a_list, path_c, 2),
        batch_size=16,
        num_workers=2,
        shuffle=True
    )

    return data_loader