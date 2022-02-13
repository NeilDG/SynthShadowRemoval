import os.path

import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional
import torch.nn.functional as F
import constants
import kornia

class ColorTransferDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, path_segment, transform_config):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.path_segment = path_segment
        self.transform_config = transform_config

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if(transform_config == 1):
            self.final_transform_op = transforms.Compose([
                # transforms.RandomCrop(constants.PATCH_IMAGE_SIZE),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_id =  self.path_segment + "/" + file_name
        #print("Img Id: ", img_id)
        img_segment = cv2.cvtColor(cv2.imread(img_id), cv2.COLOR_BGR2RGB)
        img_segment = cv2.resize(img_segment, (256, 256))

        # convert img_b to mask
        img_mask_c = cv2.inRange(img_segment[:, :, 1], 200, 255)  # green segmentation mask = road
        img_mask_c = cv2.cvtColor(img_mask_c, cv2.COLOR_GRAY2RGB)

        img_mask_d = cv2.inRange(img_segment[:, :, 0], 200, 255)  # red segmentation mask = building
        img_mask_d = cv2.cvtColor(img_mask_d, cv2.COLOR_GRAY2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        img_mask_c = self.initial_op(img_mask_c)
        img_mask_d = self.initial_op(img_mask_d)

        if(self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=constants.PATCH_IMAGE_SIZE)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            img_mask_c = transforms.functional.crop(img_mask_c, i, j, h, w)
            img_mask_d = transforms.functional.crop(img_mask_d, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
        img_mask_c = self.mask_op(img_mask_c)
        img_mask_d = self.mask_op(img_mask_d)

        return file_name, img_a, img_b, img_mask_c, img_mask_d

    def __len__(self):
        return len(self.image_list_a)

class MapDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, transform_config, opts):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if(transform_config == 1):
            self.final_transform_op = transforms.Compose([
                # transforms.RandomCrop(constants.PATCH_IMAGE_SIZE),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        file_name = os.path.basename(img_id)

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        # print("Img id: ", img_id)
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_mask = cv2.inRange(img_b[:, :, 1], 254, 255)  # mask for getting zero pixels to be excluded
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        img_mask = self.initial_op(img_mask)
        
        if(self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            img_mask = transforms.functional.crop(img_mask, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
        img_mask = 1 - self.mask_op(img_mask)

        return file_name, img_a, img_b, img_mask

    def __len__(self):
        return len(self.image_list_a)

class SmoothnessMapData():
    def create_mask_segments(self, img):
        self.img = img
        self.img_segments = [None, None, None, None, None]

        self.img_segments[0] = cv2.inRange(self.img[:, :, 1], 0, 15)  # getting the mask of specific regions of colors. Multiplier is the class label
        self.img_segments[0] = cv2.normalize(self.img_segments[0], dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.img_segments[1] = cv2.inRange(self.img[:, :, 1], 200, 255)
        self.img_segments[1] = cv2.normalize(self.img_segments[1], dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.img_segments[2] = cv2.inRange(self.img[:, :, 1], 15, 29)
        self.img_segments[2] = cv2.normalize(self.img_segments[2], dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.img_segments[3] = cv2.inRange(self.img[:, :, 1], 30, 51)
        self.img_segments[3] = cv2.normalize(self.img_segments[3], dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.img_segments[4] = cv2.inRange(self.img[:, :, 1], 52, 199)
        self.img_segments[4] = cv2.normalize(self.img_segments[4], dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return self.img_segments

    def crop_img_segments(self, i, j, h, w):
        for i in range(0, len(self.img_segments)):
            self.img_segments[i] = self.img_segments[i][i: i + h, j: j + w]

        return self.img_segments

class SpecularMapData(SmoothnessMapData):
    def create_mask_segments(self, img):
        self.img = img
        self.img_segments = [None, None, None, None]

        return self.img_segments


class RenderSegmentDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, transform_config):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.transform_config = transform_config

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if (transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

        #load one sample
        img_id = self.image_list_a[0]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]
        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id)

        self.smoothness_data = SmoothnessMapData()

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_segments = self.smoothness_data.create_mask_segments(img_b)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        #img_one_hot = self.initial_op(img_one_hot)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=constants.PATCH_IMAGE_SIZE)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            # img_one_hot = transforms.functional.crop(img_one_hot, i, j, h, w)

            # img_one_hot = img_one_hot[i: i + h, j: j + w]
            img_segments = self.smoothness_data.crop_img_segments(i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)

        img_one_hot = torch.tensor(np.asarray(img_segments), dtype = torch.float32)
        return file_name, img_a, img_one_hot

    def __len__(self):
        return len(self.image_list_a)


class RenderDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, path_c, path_d, path_e, path_f, transform_config):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.path_c = path_c
        self.path_d = path_d
        self.path_e = path_e
        self.path_f = path_f

        self.transform_config = transform_config

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if(transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


        self.smoothness_data = SmoothnessMapData()

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_id = self.path_c + "/" + file_name
        img_c = cv2.imread(img_id);
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

        img_id = self.path_d + "/" + file_name
        img_d = cv2.imread(img_id);
        img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

        img_id = self.path_e + "/" + file_name
        img_e = cv2.imread(img_id);
        img_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)

        img_id = self.path_f + "/" + file_name
        img_f = cv2.imread(img_id);
        img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        img_c = self.initial_op(img_c)
        img_d = self.initial_op(img_d)
        # img_e = self.initial_op(img_e)
        img_segments_e = self.smoothness_data.create_mask_segments(img_e)
        img_f = self.initial_op(img_f)

        if(self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=constants.PATCH_IMAGE_SIZE)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            img_c = transforms.functional.crop(img_c, i, j, h, w)
            img_d = transforms.functional.crop(img_d, i, j, h, w)
            # img_e = transforms.functional.crop(img_e, i, j, h, w)
            img_segments_e = self.smoothness_data.crop_img_segments(i, j, h, w)
            img_f = transforms.functional.crop(img_f, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
        img_c = self.final_transform_op(img_c)
        img_d = self.final_transform_op(img_d)
        # img_e = self.final_transform_op(img_e)
        img_e = torch.tensor([img_segments_e[0], img_segments_e[1], img_segments_e[2], img_segments_e[3]], dtype=torch.float32)
        img_f = self.final_transform_op(img_f)

        return file_name, img_a, img_b, img_c, img_d, img_e, img_f

    def __len__(self):
        return len(self.image_list_a)


class ShadingDataset(data.Dataset):
    def __init__(self, image_list_a, path_b, transform_config, opts):
        self.image_list_a = image_list_a
        self.path_b = path_b
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if (transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        file_name = os.path.basename(img_id)

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = self.path_b + "/" + file_name
        img_b = cv2.imread(img_id);
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)

        return file_name, img_a, img_b

    def __len__(self):
        return len(self.image_list_a)

class ShadowMapDataset(data.Dataset):
    def __init__(self, image_list_a, folder_b, folder_c, transform_config, return_shading: bool, opts):
        self.image_list_a = image_list_a
        self.folder_b = folder_b
        self.folder_c = folder_c
        self.transform_config = transform_config
        self.return_shading = return_shading
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        self.normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize_op_grey = transforms.Normalize((0.5), (0.5))

        if (transform_config == 1):
            self.final_transform_op = transforms.Compose([
                transforms.ToTensor()
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

            self.mask_op = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        file_name = os.path.basename(img_id)
        base_path = os.path.split(os.path.split(img_id)[0])[0]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_id = base_path + "/" + self.folder_b + "/" + file_name
        img_b = cv2.imread(img_id)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        img_id = constants.DATASET_PREFIX_5_PATH + self.folder_c + "/" + file_name
        img_c = cv2.imread(img_id)
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)
        img_c = self.initial_op(img_c)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices

            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)
            img_c = transforms.functional.crop(img_c, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_b = self.final_transform_op(img_b)
        img_c = self.final_transform_op(img_c)

        img_a = self.normalize_op(img_a)
        img_b = self.normalize_op_grey(img_b)
        img_c = self.normalize_op(img_c)

        if(self.return_shading):
            return file_name, img_a, img_b, img_c
        else:
            return file_name, img_a, img_b

    def __len__(self):
        return len(self.image_list_a)

class RealWorldDataset(data.Dataset):
    def __init__(self, image_list_a):
        self.image_list_a = image_list_a

        self.initial_op = transforms.Compose([
            transforms.ToPILImage()])


        self.final_transform_op = transforms.Compose([
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_a = self.initial_op(img_a)
        img_a = self.final_transform_op(img_a)

        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)

class RealWorldTrainDataset(data.Dataset):
    def __init__(self, image_list_a):
        self.image_list_a = image_list_a

        self.initial_op = transforms.Compose([
            transforms.ToPILImage()])


        self.final_transform_op = transforms.Compose([
            transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.RandomCrop(constants.PATCH_IMAGE_SIZE),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img_id = self.image_list_a[idx]
        path_segment = img_id.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)  # because matplot uses RGB, openCV is BGR

        img_a = self.initial_op(img_a)
        img_a = self.final_transform_op(img_a)

        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)