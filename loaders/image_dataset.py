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

def normalize(light_angle):
    std = light_angle / 360.0
    min = -1.0
    max = 1.0
    scaled = std * (max - min) + min

    return scaled

class GenericPairedDataset(data.Dataset):
    def __init__(self, imgx_dir, img_ydir, transform_config, opts):
        self.imgx_dir = imgx_dir
        self.imgy_dir = img_ydir
        self.transform_config = transform_config
        self.patch_size = (opts.patch_size, opts.patch_size)

        if(transform_config == 1):
            # self.transform_op = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((256, 256)),
            #     transforms.RandomCrop(self.patch_size),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])

            self.transform_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        else:
            self.transform_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, idx):
        img_x_path = self.imgx_dir[idx]
        img_x = cv2.imread(img_x_path)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        img_y_path = self.imgy_dir[idx % len(self.imgy_dir)]
        img_y = cv2.imread(img_y_path)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)

        tensor_x = self.transform_op(img_x)
        tensor_y = self.transform_op(img_y)

        return tensor_x, tensor_y

    def __len__(self):
        return len(self.imgx_dir)

class RelightDataset(data.Dataset):
    def __init__(self, img_length, rgb_dir, albedo_dir, scene_names, opts):
        self.img_length = img_length
        self.rgb_dir = rgb_dir
        self.albedo_dir = albedo_dir
        self.scene_names = scene_names
        self.patch_size = (opts.patch_size, opts.patch_size)

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        file_name = "/synth_" + str(idx) + ".png"

        scene_index = np.random.randint(0, len(self.scene_names))
        rgb_path = self.rgb_dir + self.scene_names[scene_index] + file_name

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.imread(self.albedo_dir + file_name)
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)

        rgb_tensor = self.initial_op(rgb_img)
        albedo_tensor = self.initial_op(albedo_img)

        scene_tensor = (scene_index / len(self.scene_names)) * 1.0 #normalize to 0.0 - 1.0
        scene_tensor = torch.tensor(scene_tensor)

        return file_name, rgb_tensor, albedo_tensor, scene_tensor

    def __len__(self):
        return self.img_length



class IIDDataset(data.Dataset):
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
        if(idx == 9586):
            idx = np.random.randint(0, 9585) #TEMP FIX for missing PNG file

        file_name = "synth_" + str(idx) + ".png"

        img_a_path = self.albedo_dir + file_name
        albedo = cv2.imread(img_a_path) #albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        img_b_path = self.shading_dir + file_name
        shading = cv2.imread(img_b_path) #shading
        shading = cv2.cvtColor(shading, cv2.COLOR_BGR2GRAY)

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
        shading = self.normalize_op_grey(shading)
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


class GTATestDataset(data.Dataset):
    def __init__(self, rgb_list, albedo_list, opts):
        self.rgb_list = rgb_list
        self.albedo_list = albedo_list
        self.patch_size = (opts.patch_size, opts.patch_size)
        self.light_angles = [0, 36, 72, 108, 144]

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))])

        self.normalize_op = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.normalize_op_grey = transforms.Normalize((0.5), (0.5))

        self.final_transform_op = transforms.Compose([
            # transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()
        ])

        self.mask_op = transforms.Compose([
            # transforms.Resize(constants.TEST_IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # file_name = "synth_" + str(idx) + ".png"

        albedo = cv2.imread(self.albedo_list[idx])  # albedo
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)

        input_rgb = cv2.imread(self.rgb_list[idx])  # input rgb
        input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2RGB)

        input_rgb = self.initial_op(input_rgb)
        albedo = self.initial_op(albedo)

        input_rgb = self.final_transform_op(input_rgb)
        albedo = self.final_transform_op(albedo)

        input_rgb = self.normalize_op(input_rgb)
        albedo = self.normalize_op(albedo)

        return input_rgb, albedo

    def __len__(self):
        return len(self.rgb_list)

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