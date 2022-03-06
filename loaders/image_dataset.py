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

def normalize(light_angle):
    std = light_angle / 360.0
    min = -1.0
    max = 1.0
    scaled = std * (max - min) + min

    return scaled

class ImageRelightDataset(data.Dataset):
    def __init__(self, img_length, rgb_dir, albedo_dir, shading_dir, shadow_dir, transform_config, opts):
        self.img_length = img_length
        self.albedo_dir = albedo_dir
        self.shading_dir = shading_dir
        self.shadow_dir = shadow_dir
        self.rgb_dir = rgb_dir
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)
        self.light_angles = [0, 36, 72, 108, 144]

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
        file_name = "synth_" + str(idx) + ".png"

        img_a_path = self.albedo_dir + file_name
        albedo = cv2.imread(img_a_path) #albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        img_b_path = self.shading_dir + file_name
        shading = cv2.imread(img_b_path) #shading
        shading = cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)

        #randomize light angle
        light_angle_a = np.random.choice(self.light_angles)
        # light_angle_a = 36
        img_rgb_path = self.rgb_dir.format(input_light_angle=light_angle_a) + file_name
        input_rgb = cv2.imread(img_rgb_path)  # input rgb
        input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2RGB)

        img_c_path = self.shadow_dir.format(input_light_angle=light_angle_a) + file_name
        input_shadow_map = cv2.imread(img_c_path)  # target shadow map
        input_shadow_map = cv2.cvtColor(input_shadow_map, cv2.COLOR_BGR2GRAY)

        #randomize light_angle
        light_angle_b = np.random.choice(self.light_angles)
        while(light_angle_b == light_angle_a):
            light_angle_b = np.random.choice(self.light_angles)
        # light_angle_b = 144
        img_c_path = self.shadow_dir.format(input_light_angle=light_angle_b) + file_name
        target_shadow_map = cv2.imread(img_c_path) #target shadow map
        target_shadow_map = cv2.cvtColor(target_shadow_map, cv2.COLOR_BGR2GRAY)

        img_d_path = self.rgb_dir.format(input_light_angle=light_angle_b) + file_name
        target_rgb = cv2.imread(img_d_path) #target rgb
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        input_rgb = self.initial_op(input_rgb)
        albedo = self.initial_op(albedo)
        shading = self.initial_op(shading)
        input_shadow_map = self.initial_op(input_shadow_map)
        target_shadow_map = self.initial_op(target_shadow_map)
        target_rgb = self.initial_op(target_rgb)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(shading, output_size=self.patch_size)
            i, j, h, w = crop_indices

            input_rgb = transforms.functional.crop(input_rgb, i, j, h, w)
            albedo = transforms.functional.crop(albedo, i, j, h, w)
            shading = transforms.functional.crop(shading, i, j, h, w)
            input_shadow_map = transforms.functional.crop(input_shadow_map, i, j, h, w)
            target_shadow_map = transforms.functional.crop(target_shadow_map, i, j, h, w)
            target_rgb = transforms.functional.crop(target_rgb, i, j, h, w)

        input_rgb = self.final_transform_op(input_rgb)
        albedo = self.final_transform_op(albedo)
        shading = self.final_transform_op(shading)
        input_shadow_map = self.final_transform_op(input_shadow_map)
        target_shadow_map = self.final_transform_op(target_shadow_map)
        target_rgb = self.final_transform_op(target_rgb)

        input_rgb = self.normalize_op(input_rgb)
        albedo = self.normalize_op(albedo)
        shading = self.normalize_op(shading)
        input_shadow_map = self.normalize_op_grey(input_shadow_map)
        target_shadow_map = self.normalize_op_grey(target_shadow_map)
        target_rgb = self.normalize_op(target_rgb)

        light_angle_b = normalize(light_angle_b)
        light_angle_tensor = torch.full_like(target_shadow_map[:, :, :], light_angle_b)

        return file_name, input_rgb, albedo, shading, input_shadow_map, target_shadow_map, target_rgb, light_angle_tensor

    def __len__(self):
        return self.img_length

class ShadowRelightDatset(data.Dataset):
    def __init__(self, img_length, input_rgb_path, input_shadow_path, desired_shadow_path, transform_config, opts):
        self.img_length = img_length
        self.path_a = input_shadow_path
        self.path_b = desired_shadow_path
        self.path_rgb = input_rgb_path
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)
        self.light_angles = [0, 36, 72, 108, 144]

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        if (transform_config == 1):
            self.final_transform_op_grey = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            self.final_transform_op = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.mask_op = transforms.Compose([
                transforms.ToTensor()
            ])

        else:
            self.final_transform_op_grey = transforms.Compose([
                transforms.Resize(constants.TEST_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

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
        # img_id = self.image_list_a[idx]
        file_name = "synth_"+ str(idx) + ".png"

        light_angle_a = np.random.choice(self.light_angles)
        img_a_path = self.path_a.format(input_light_angle=light_angle_a) + file_name
        img_a = cv2.imread(img_a_path)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)

        path = self.path_rgb.format(input_light_angle=light_angle_a) + file_name
        img_rgb = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        #randomize desired light angle
        light_angle_b = np.random.choice(self.light_angles)
        while light_angle_a == light_angle_b:
            light_angle_b = np.random.choice(self.light_angles)
        img_b_path = self.path_b.format(input_light_angle = light_angle_b) + file_name
        # print("Img path pairing: ", img_a_path, img_b_path)

        img_b = cv2.imread(img_b_path)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        img_rgb = self.initial_op(img_rgb)
        img_a = self.initial_op(img_a)
        img_b = self.initial_op(img_b)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices

            img_rgb = transforms.functional.crop(img_rgb, i, j, h, w)
            img_a = transforms.functional.crop(img_a, i, j, h, w)
            img_b = transforms.functional.crop(img_b, i, j, h, w)

        img_rgb = self.final_transform_op(img_rgb)
        img_a = self.final_transform_op_grey(img_a)
        img_b = self.final_transform_op_grey(img_b)
        light_angle_b = normalize(light_angle_b)
        light_angle_tensor = torch.full_like(img_a[:, :, :], light_angle_b)

        return file_name, img_rgb, img_a, img_b, light_angle_tensor

    def __len__(self):
        return self.img_length

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

class ShadowPriorDataset(data.Dataset):
    def __init__(self, image_list_a, transform_config, opts):
        self.image_list_a = image_list_a
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

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

        # print(img_id)
        img_a = cv2.imread(img_id);
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)  # because matplot uses RGB, openCV is BGR
        img_a = self.initial_op(img_a)

        if (self.transform_config == 1):
            crop_indices = transforms.RandomCrop.get_params(img_a, output_size=self.patch_size)
            i, j, h, w = crop_indices
            img_a = transforms.functional.crop(img_a, i, j, h, w)

        img_a = self.final_transform_op(img_a)
        img_a = self.normalize_op_grey(img_a)
        return file_name, img_a

    def __len__(self):
        return len(self.image_list_a)