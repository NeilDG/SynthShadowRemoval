import glob
import sys

from loaders import dataset_loader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import tensor_utils
from skimage.metrics import structural_similarity as ssim
from optparse import OptionParser
import kornia
import torch
from torch.nn import functional as F
from utils import plot_utils

parser = OptionParser()
parser.add_option('--shading_multiplier', type=float, default=1.0)
parser.add_option('--shadow_multiplier', type=float, default=1.0)

def test_lighting():
    RGB_PATH = "E:/SynthWeather Dataset 2/default/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 2/albedo/"
    rgb_list = dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)

    for i, (rgb_path, albedo_path) \
            in enumerate(zip(rgb_list, albedo_list)):
        path_segment = rgb_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        albedo_img = cv2.imread(albedo_path)
        rgb_img = cv2.imread(rgb_path)

        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_mask = cv2.inRange(albedo_img[:, :, 0], 0.0, 0.1)  # mask for getting zero pixels to be excluded
        img_mask = np.asarray([img_mask, img_mask, img_mask])
        img_mask = np.moveaxis(img_mask, 0, 2)

        img_ones = np.full_like(img_mask, 1.0)

        # albedo_img = np.clip(albedo_img + (rgb_img * img_mask), 0.01, 1.0) #add skybox
        albedo_img = np.clip(albedo_img + (img_ones * img_mask), 0.01, 1.0)
        light_color = np.asarray([225, 247, 250]) / 255.0

        # shading_component = np.clip((rgb_img / albedo_img) - lightmap_img , 0.0, 1.0)
        # rgb_img_like = np.clip((albedo_img * shading_component + lightmap_img), 0.0, 1.0)

        # shading_component = np.clip((rgb_img / albedo_img), 0.0, 1.0)
        # rgb_img_like = np.clip((albedo_img * shading_component), 0.0, 1.0)

        rgb_img_like = np.full_like(albedo_img, 0.0)
        shading_component = np.clip((rgb_img / albedo_img), 0.0, 1.0)
        shading_component[:, :, 0] = shading_component[:, :, 0] / light_color[0]
        shading_component[:, :, 1] = shading_component[:, :, 1] / light_color[1]
        shading_component[:, :, 2] = shading_component[:, :, 2] / light_color[2]

        light_color = np.asarray([np.random.randn(), np.random.randn(), np.random.randn()])
        rgb_img_like[:, :, 0] = np.clip((albedo_img[:, :, 0] * shading_component[:, :, 0] * light_color[0]), 0.0, 1.0)
        rgb_img_like[:, :, 1] = np.clip((albedo_img[:, :, 1] * shading_component[:, :, 1] * light_color[1]), 0.0, 1.0)
        rgb_img_like[:, :, 2] = np.clip((albedo_img[:, :, 2] * shading_component[:, :, 2] * light_color[2]), 0.0, 1.0)
        diff = rgb_img - rgb_img_like
        print("Difference: ", np.mean(diff))

        plt.imshow(albedo_img)
        plt.show()

        plt.imshow(shading_component)
        plt.show()

        plt.imshow(rgb_img_like)
        plt.show()

        plt.imshow(rgb_img)
        plt.show()

        break

        # cv2.imwrite("E:/SynthWeather Dataset 3/rgb/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img_like, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 3/shading/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shading_component, alpha=255.0), cv2.COLOR_BGR2RGB))
        # # cv2.imwrite("E:/SynthWeather Dataset 3/lightmap_img/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(lightmap_img, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 3/albedo/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(albedo_img, alpha=255.0), cv2.COLOR_BGR2RGB))

def test_deferred_render():
    RGB_PATH = "E:/SynthWeather Dataset 2/default/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 2/albedo/"
    POSITION_PATH = "E:/SynthWeather Dataset 2/position/"
    NORMAL_PATH = "E:/SynthWeather Dataset 2/normal/"
    SPECULAR_PATH = "E:/SynthWeather Dataset 2/specular/"
    SMOOTHNESS_PATH = "E:/SynthWeather Dataset 2/smoothness/"
    LIGHTMAP_PATH = "E:/SynthWeather Dataset 2/lightmap/"

    rgb_list = dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)
    position_list = dataset_loader.assemble_unpaired_data(POSITION_PATH, -1)
    normal_list = dataset_loader.assemble_unpaired_data(NORMAL_PATH, -1)
    specular_list = dataset_loader.assemble_unpaired_data(SPECULAR_PATH, -1)
    smoothness_list = dataset_loader.assemble_unpaired_data(SMOOTHNESS_PATH, -1)
    lightness_list = dataset_loader.assemble_unpaired_data(LIGHTMAP_PATH, -1)

    for i, (rgb_path, albedo_path, position_path, normal_path, specular_path, smoothness_path, lightmap_path) \
            in enumerate(zip(rgb_list, albedo_list, position_list, normal_list, specular_list, smoothness_list, lightness_list)):
        path_segment = rgb_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        albedo_img = cv2.imread(albedo_path)
        position_img = cv2.imread(position_path)
        normal_img = cv2.imread(normal_path)
        specular_img = cv2.imread(specular_path)
        smoothness_img = cv2.imread(smoothness_path)
        lightmap_img = cv2.imread(lightmap_path)
        rgb_img = cv2.imread(rgb_path)

        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
        position_img = cv2.cvtColor(position_img, cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        specular_img = cv2.cvtColor(specular_img, cv2.COLOR_BGR2RGB)
        smoothness_img = cv2.cvtColor(smoothness_img, cv2.COLOR_BGR2RGB)
        lightmap_img = cv2.cvtColor(lightmap_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        position_img = cv2.normalize(position_img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normal_img = cv2.normalize(normal_img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        specular_img = cv2.normalize(specular_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        smoothness_img = cv2.normalize(smoothness_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lightmap_img = cv2.normalize(lightmap_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img_mask = cv2.inRange(albedo_img[:, :, 0], 0.0, 0.1)  # mask for getting zero pixels to be excluded
        img_mask = np.asarray([img_mask, img_mask, img_mask])
        img_mask = np.moveaxis(img_mask, 0, 2)

        img_ones = np.full_like(img_mask, 1.0)

        # albedo_img = np.clip(albedo_img + (rgb_img * img_mask), 0.01, 1.0) #add skybox
        albedo_img = np.clip(albedo_img + (img_ones * img_mask), 0.01, 1.0)

        light_color = np.asarray([225, 255, 255]) / 255.0
        # light_color = np.asarray([np.random.randn(), np.random.randn(), np.random.randn()])

        light_transform = np.asarray([np.full_like(position_img[:, :, 0], 0.2),
                                     np.full_like(position_img[:, :, 1], 0.0),
                                     np.full_like(position_img[:, :, 2], 0.0)])
        light_transform = np.moveaxis(light_transform, 0, 2)

        light_dir = tensor_utils.normalize(light_transform - position_img)
        diffuse = np.clip(np.vdot(normal_img, light_dir), 0.0, 1.0) * albedo_img * light_color
        rgb_img_like = albedo_img + diffuse

        plt.imshow(rgb_img_like)
        plt.show()

        plt.imshow(rgb_img)
        plt.show()

        # cv2.imwrite("E:/SynthWeather Dataset 3/rgb/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img_like, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 3/shading/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shading_component, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 3/shadow_map/" + file_name, cv2.convertScaleAbs(shadow_map, alpha=255.0))
        # cv2.imwrite("E:/SynthWeather Dataset 3/albedo/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(albedo_img, alpha=255.0), cv2.COLOR_BGR2RGB))

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img

def compute_and_produce_rgb_v1(type_prefix, degree_prefix, argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    RGB_PATH = "E:/SynthWeather Dataset 5 - RAW/" + type_prefix + "/" + degree_prefix + "/rgb/"
    print("RGB path: ", RGB_PATH)
    RGB_NOSHADOWS_PATH = "E:/SynthWeather Dataset 5 - RAW/no_shadows/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 5/albedo/"

    rgb_list = dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    noshadows_list = dataset_loader.assemble_unpaired_data(RGB_NOSHADOWS_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)

    shadows_diff_mean = 0.0
    for i, (rgb_path, noshadows_path, albedo_path) \
            in enumerate(zip(rgb_list, noshadows_list, albedo_list)):
        path_segment = rgb_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        rgb_img = cv2.imread(rgb_path)
        noshadows_img = cv2.imread(noshadows_path)
        albedo_img = cv2.imread(albedo_path)

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        noshadows_img = cv2.cvtColor(noshadows_img, cv2.COLOR_BGR2RGB)
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)

        rgb_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        noshadows_img = cv2.normalize(noshadows_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # img_mask = cv2.inRange(albedo_img[:, :, 0], 0.0, 0.1)  # mask for getting zero pixels to be excluded
        # img_mask = np.asarray([img_mask, img_mask, img_mask])
        # img_mask = np.moveaxis(img_mask, 0, 2)
        #
        # img_ones = np.full_like(img_mask, 1.0)
        #
        # #extract shading component
        # albedo_img = np.clip(albedo_img + (img_ones * img_mask), 0.01, 1.0)
        albedo_img = np.clip(albedo_img, 0.00001, 1.0)
        light_color = np.asarray([255, 255, 255]) / 255.0 #values are extracted from Unity Engine
        # light_color = light_color * 1.5

        #extract shadows
        noshadows_img = np.clip(noshadows_img, 0.00001, 1.0)
        shadow_map = np.clip(rgb_img  / noshadows_img, 0.00001, 1.0)

        # compress shadow map since varying information across all channels are very minute
        shadow_map = (shadow_map[:, :, 0] + shadow_map[:, :, 1] + shadow_map[:, :, 2]) / 3.0
        shadow_map = shadow_map * opts.shadow_multiplier
        shadow_map = np.clip(shadow_map, 0.00001, 1.0)

        shading_component = np.full_like(albedo_img, 0.0)
        shading_component[:, :, 0] = noshadows_img[:, :, 0] / (albedo_img[:, :, 0] * light_color[0])
        shading_component[:, :, 1] = noshadows_img[:, :, 1] / (albedo_img[:, :, 1] * light_color[1])
        shading_component[:, :, 2] = noshadows_img[:, :, 2] / (albedo_img[:, :, 2] * light_color[2])
        shading_component = shading_component * opts.shading_multiplier
        shading_component = np.clip(shading_component, 0.00001, 1.0)
        # shading_component = kornia.enhance.posterize(torch.unsqueeze(torch.from_numpy(shading_component), 1), torch.tensor(2))
        # shading_component = torch.squeeze(shading_component).numpy()

        # light_color = np.asarray([np.random.randn() + 0.25, np.random.randn() + 0.25, np.random.randn() + 0.25])
        rgb_img_like = np.full_like(albedo_img, 0.0)
        rgb_img_like[:, :, 0] = np.clip((albedo_img[:, :, 0] * shading_component[:, :, 0] * light_color[0]) * shadow_map, 0.0, 1.0)
        rgb_img_like[:, :, 1] = np.clip((albedo_img[:, :, 1] * shading_component[:, :, 1] * light_color[1]) * shadow_map, 0.0, 1.0)
        rgb_img_like[:, :, 2] = np.clip((albedo_img[:, :, 2] * shading_component[:, :, 2] * light_color[2]) * shadow_map, 0.0, 1.0)
        # rgb_img_like[:, :, 0] = np.clip((albedo_img[:, :, 0] * shading_component[:, :, 0] * light_color[0]), 0.0, 1.0)
        # rgb_img_like[:, :, 1] = np.clip((albedo_img[:, :, 1] * shading_component[:, :, 1] * light_color[1]), 0.0, 1.0)
        # rgb_img_like[:, :, 2] = np.clip((albedo_img[:, :, 2] * shading_component[:, :, 2] * light_color[2]), 0.0, 1.0)
        diff = rgb_img - rgb_img_like
        print("Difference: ", np.mean(diff))

        # rgb_closed_form = np.asarray([rgb_img_like[:, :, 0] * shadow_map,
        #                               rgb_img_like[:, :, 1] * shadow_map,
        #                               rgb_img_like[:, :, 2] * shadow_map])
        # rgb_closed_form = np.moveaxis(rgb_closed_form, 0, 2)

        plt.imshow(rgb_img)
        plt.show()

        plt.imshow(shading_component)
        plt.show()

        plt.imshow(shadow_map, cmap='gray')
        plt.show()

        plt.imshow(rgb_img_like)
        plt.show()
        break

        # cv2.imwrite("E:/SynthWeather Dataset 5/" + type_prefix + "/" + degree_prefix + "/rgb/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img_like, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 5/" + type_prefix + "/" + degree_prefix + "/shading/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shading_component, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 5/shading/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shading_component, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 5/" + type_prefix + "/" + degree_prefix + "/shadow_map/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shadow_map, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 5/" + type_prefix + "/" + degree_prefix + "/shadow_map/" + file_name, cv2.convertScaleAbs(shadow_map, alpha=255.0))
        # # cv2.imwrite("E:/SynthWeather Dataset 5/" + degree_prefix + "/albedo/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(albedo_img, alpha=255.0), cv2.COLOR_BGR2RGB))

def compute_and_produce_rgb_v2(type_prefix, degree_prefix, argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    RGB_PATH = "E:/SynthWeather Dataset 5 - RAW/" + type_prefix + "/" + degree_prefix + "/rgb/"
    print("RGB path: ", RGB_PATH)
    RGB_NOSHADOWS_PATH = "E:/SynthWeather Dataset 5 - RAW/no_shadows/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 5/albedo/"

    rgb_list = dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    noshadows_list = dataset_loader.assemble_unpaired_data(RGB_NOSHADOWS_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)

    shadows_diff_mean = 0.0
    for i, (rgb_path, noshadows_path, albedo_path) \
            in enumerate(zip(rgb_list, noshadows_list, albedo_list)):
        path_segment = rgb_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        rgb_img = cv2.imread(rgb_path)
        noshadows_img = cv2.imread(noshadows_path)
        albedo_img = cv2.imread(albedo_path)

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        noshadows_img = cv2.cvtColor(noshadows_img, cv2.COLOR_BGR2RGB)
        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)

        rgb_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        noshadows_img = cv2.normalize(noshadows_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        albedo_img = np.clip(albedo_img, 0.00001, 1.0)
        light_color = np.asarray([255, 255, 255]) / 255.0 #values are extracted from Unity Engine
        # light_color = light_color * 1.5

        #extract shadows
        noshadows_img = np.clip(noshadows_img, 0.00001, 1.0)
        shadow_map = np.clip(rgb_img  / noshadows_img, 0.00001, 1.0)

        # compress shadow map since varying information across all channels are very minute
        # shadow_map = (shadow_map[:, :, 0] + shadow_map[:, :, 1] + shadow_map[:, :, 2]) / 3.0
        shadow_map = cv2.cvtColor(shadow_map, cv2.COLOR_BGR2GRAY)
        shadow_map = shadow_map * opts.shadow_multiplier
        shadow_map = np.clip(shadow_map, 0.00001, 1.0)


        shading_component = np.full_like(albedo_img, 0.0)
        shading_component[:, :, 0] = noshadows_img[:, :, 0] / (albedo_img[:, :, 0] * light_color[0])
        shading_component[:, :, 1] = noshadows_img[:, :, 1] / (albedo_img[:, :, 1] * light_color[1])
        shading_component[:, :, 2] = noshadows_img[:, :, 2] / (albedo_img[:, :, 2] * light_color[2])
        shading_component = shading_component * opts.shading_multiplier
        shading_component = np.clip(shading_component, 0.00001, 1.0)

        shading_component = cv2.cvtColor(shading_component, cv2.COLOR_RGB2GRAY)

        # refine albedo
        albedo_img[:, :, 0] = noshadows_img[:, :, 0] / shading_component
        albedo_img[:, :, 1] = noshadows_img[:, :, 1] / shading_component
        albedo_img[:, :, 2] = noshadows_img[:, :, 2] / shading_component
        albedo_img = np.clip(albedo_img, 0.00001, 1.0)

        # light_color = np.asarray([np.random.randn(), np.random.randn(), np.random.randn()])
        rgb_img_like = np.full_like(albedo_img, 0.0)
        rgb_img_like[:, :, 0] = np.clip((albedo_img[:, :, 0] * shading_component * light_color[0]) * shadow_map, 0.0, 1.0)
        rgb_img_like[:, :, 1] = np.clip((albedo_img[:, :, 1] * shading_component * light_color[1]) * shadow_map, 0.0, 1.0)
        rgb_img_like[:, :, 2] = np.clip((albedo_img[:, :, 2] * shading_component * light_color[2]) * shadow_map, 0.0, 1.0)

        diff = rgb_img - rgb_img_like
        print("Difference: ", np.mean(diff))

        # plt.imshow(rgb_img)
        # plt.show()
        #
        # plt.imshow(albedo_img)
        # plt.show()
        #
        # plt.imshow(shading_component, cmap='gray')
        # plt.show()
        #
        # plt.imshow(shadow_map, cmap='gray')
        # plt.show()
        #
        # plt.imshow(rgb_img_like)
        # plt.show()
        # break

        # cv2.imwrite("E:/SynthWeather Dataset 6/" + type_prefix + "/" + degree_prefix + "/rgb/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img_like, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 6/shading/" + file_name, cv2.convertScaleAbs(shading_component, alpha=255.0))
        # cv2.imwrite("E:/SynthWeather Dataset 6/" + type_prefix + "/" + degree_prefix + "/shadow_map/" + file_name, cv2.convertScaleAbs(shadow_map, alpha=255.0))
        cv2.imwrite("E:/SynthWeather Dataset 6/albedo/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(albedo_img, alpha=255.0), cv2.COLOR_BGR2RGB))


def produce_rgbs(type_prefix, degree_prefix):
    SHADOWS_PATH = "E:/SynthWeather Dataset 4/" + type_prefix + "/" + degree_prefix + "/shadow_map/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 4/albedo/"
    SHADING_PATH = "E:/SynthWeather Dataset 4/shading/"
    light_color = np.asarray([225, 247, 250]) / 255.0 #values are extracted from Unity Engine

    shadow_list = dataset_loader.assemble_unpaired_data(SHADOWS_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)
    shading_list = dataset_loader.assemble_unpaired_data(SHADING_PATH, -1)

    shadows_diff_mean = 0.0
    for i, (albedo_path, shading_path, shadow_path) \
            in enumerate(zip(albedo_list, shading_list, shadow_list)):
        path_segment = albedo_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        albedo_img = load_img(albedo_path)
        shading_img = load_img(shading_path)
        shadow_img = load_img(shadow_path)

        rgb_img = np.full_like(albedo_img, 0)
        rgb_img[:, :, 0] = np.clip(albedo_img[:, :, 0] * shading_img[:, :, 0] * light_color[0] * shadow_img[:, :, 0], 0.0, 1.0)
        rgb_img[:, :, 1] = np.clip(albedo_img[:, :, 1] * shading_img[:, :, 1] * light_color[1] * shadow_img[:, :, 1], 0.0, 1.0)
        rgb_img[:, :, 2] = np.clip(albedo_img[:, :, 2] * shading_img[:, :, 2] * light_color[2] * shadow_img[:, :, 2], 0.0, 1.0)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)

        cv2.imwrite("E:/SynthWeather Dataset 4/" + type_prefix + "/" + degree_prefix + "/rgb_new/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img, alpha=255.0), cv2.COLOR_BGR2RGB))
        print("Saved: " + "E:/SynthWeather Dataset 4/" + type_prefix + "/" + degree_prefix + "/rgb_new/" + file_name)

def measure_shading_diff(path_a, path_b):
    a_list = dataset_loader.assemble_unpaired_data(path_a, -1)
    b_list = dataset_loader.assemble_unpaired_data(path_b, -1)

    ssim_measure = 0.0
    for i, (a_path, b_path) in enumerate(zip(a_list, b_list)):
        a_img = cv2.imread(a_path)
        b_img = cv2.imread(b_path)

        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)

        a_img = cv2.normalize(a_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        b_img = cv2.normalize(b_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        ssim_measure = ssim_measure + ssim(a_img, b_img, multichannel=True)

    ssim_measure = ssim_measure / len(a_list)
    print("Average SSIM between %s %s: %f" %(a_path, b_path, ssim_measure))


def measure_performance():

    visdom_reporter = plot_utils.VisdomReporter()

    GTA_BASE_PATH = "E:IID-TestDataset/GTA/"
    RGB_PATH = GTA_BASE_PATH + "/input/"
    ALBEDO_PATH = GTA_BASE_PATH + "/albedo/"

    RESULT_A_PATH = GTA_BASE_PATH + "/li_eccv18/"
    RESULT_B_PATH = GTA_BASE_PATH + "/yu_cvpr19/"
    RESULT_C_PATH = GTA_BASE_PATH + "/yu_eccv20/"
    RESULT_D_PATH = GTA_BASE_PATH + "/zhu_iccp21/"

    rgb_list = glob.glob(RGB_PATH + "*.png")
    albedo_list = glob.glob(ALBEDO_PATH + "*.png")
    a_list = glob.glob(RESULT_A_PATH + "*.png")
    b_list = glob.glob(RESULT_B_PATH + "*.png")
    c_list = glob.glob(RESULT_C_PATH + "*.png")
    d_list = glob.glob(RESULT_D_PATH + "*.png")

    IMG_SIZE = (320, 240)

    albedo_tensor = tensor_utils.load_metric_compatible_img(albedo_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    a_tensor = tensor_utils.load_metric_compatible_img(a_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    b_tensor = tensor_utils.load_metric_compatible_img(b_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    c_tensor = tensor_utils.load_metric_compatible_img(c_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
    d_tensor = tensor_utils.load_metric_compatible_img(d_list[0], cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)

    for i, (rgb_path, albedo_path, a_path, b_path, c_path, d_path) in enumerate(zip(rgb_list, albedo_list, a_list, b_list, c_list, d_list)):
        rgb_img = tensor_utils.load_metric_compatible_img(rgb_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)
        albedo_img = tensor_utils.load_metric_compatible_img(albedo_path, cv2.COLOR_BGR2RGB, True, False, IMG_SIZE)

        #compute shading
        # img_mask = cv2.inRange(albedo_img[:, :, 0], 0.0, 0.1)  # mask for getting zero pixels to be excluded
        # img_mask = np.asarray([img_mask, img_mask, img_mask])
        # img_mask = np.moveaxis(img_mask, 0, 2)
        #
        # img_ones = np.full_like(img_mask, 1.0)
        #
        # # albedo_img = np.clip(albedo_img + (rgb_img * img_mask), 0.01, 1.0) #add skybox
        # albedo_img = np.clip(albedo_img + (img_ones * img_mask), 0.01, 1.0)
        # shading_img = cv2.cvtColor(rgb_img / albedo_img, cv2.COLOR_BGR2GRAY)
        # shading_img = np.clip(shading_img, 0.00001, 1.0)

        #refine albedo gt
        # albedo_img[:, :, 0] = rgb_img[:, :, 0] / shading_img
        # albedo_img[:, :, 1] = rgb_img[:, :, 1] / shading_img
        # albedo_img[:, :, 2] = rgb_img[:, :, 2] / shading_img

        # rgb_closed_form = np.full_like(rgb_img, 0)
        # rgb_closed_form[:, :, 0] = np.clip(albedo_img[:, :, 0] * shading_img, 0.0, 1.0)
        # rgb_closed_form[:, :, 1] = np.clip(albedo_img[:, :, 1] * shading_img, 0.0, 1.0)
        # rgb_closed_form[:, :, 2] = np.clip(albedo_img[:, :, 2] * shading_img, 0.0, 1.0)

        albedo_img = tensor_utils.load_metric_compatible_img(albedo_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        a_img = tensor_utils.load_metric_compatible_img(a_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        b_img = tensor_utils.load_metric_compatible_img(b_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        c_img = tensor_utils.load_metric_compatible_img(c_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)
        d_img = tensor_utils.load_metric_compatible_img(d_path, cv2.COLOR_BGR2RGB, True, True, IMG_SIZE)

        psnr_albedo_a = np.round(kornia.metrics.psnr(a_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(a_img, albedo_img, 5).item(), 4)
        psnr_albedo_b = np.round(kornia.metrics.psnr(b_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(b_img, albedo_img, 5).item(), 4)
        psnr_albedo_c = np.round(kornia.metrics.psnr(c_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(c_img, albedo_img, 5).item(), 4)
        psnr_albedo_d = np.round(kornia.metrics.psnr(d_img, albedo_img, max_val=1.0).item(), 4)
        ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(d_img, albedo_img, 5).item(), 4)
        display_text = "Image " +str(i)+ " Albedo PSNR: " + str(psnr_albedo_a) + "<br> Albedo SSIM: " + str(ssim_albedo_a) + "<br>" \
                                     "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                                     "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                                     "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                                     "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>"

        visdom_reporter.plot_text(display_text)

        albedo_tensor = torch.cat([albedo_tensor, albedo_img], 0)
        a_tensor = torch.cat([a_tensor, a_img], 0)
        b_tensor = torch.cat([b_tensor, b_img], 0)
        c_tensor = torch.cat([c_tensor, c_img], 0)
        d_tensor = torch.cat([d_tensor, d_img], 0)

    print(np.shape(albedo_tensor), np.shape(a_tensor), np.shape(c_tensor), np.shape(d_tensor))
    psnr_albedo_a = np.round(kornia.metrics.psnr(a_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_a = np.round(1.0 - kornia.losses.ssim_loss(a_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_b = np.round(kornia.metrics.psnr(b_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_b = np.round(1.0 - kornia.losses.ssim_loss(b_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_c = np.round(kornia.metrics.psnr(c_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_c = np.round(1.0 - kornia.losses.ssim_loss(c_tensor, albedo_tensor, 5).item(), 4)
    psnr_albedo_d = np.round(kornia.metrics.psnr(d_tensor, albedo_tensor, max_val=1.0).item(), 4)
    ssim_albedo_d = np.round(1.0 - kornia.losses.ssim_loss(d_tensor, albedo_tensor, 5).item(), 4)
    display_text = "Image " +str(i)+ " Albedo PSNR: " + str(psnr_albedo_a) + "<br> Albedo SSIM: " + str(ssim_albedo_a) + "<br>" \
                                 "li_eccv18 PSNR: " + str(psnr_albedo_a) + "<br> SSIM: " + str(ssim_albedo_a) + "<br>" \
                                 "yu_cvpr19 PSNR: " + str(psnr_albedo_b) + "<br> SSIM: " + str(ssim_albedo_b) + "<br>" \
                                 "yu_eccv20 PSNR: " + str(psnr_albedo_c) + "<br> SSIM: " + str(ssim_albedo_c) + "<br>" \
                                 "zhu_iccp21 PSNR: " + str(psnr_albedo_d) + "<br> SSIM: " + str(ssim_albedo_d) + "<br>"

    visdom_reporter.plot_text(display_text)

    visdom_reporter.plot_image(albedo_tensor, "Albedo GT")
    visdom_reporter.plot_image(a_tensor, "li_eccv18")
    visdom_reporter.plot_image(b_tensor, "yu_cvpr19")
    visdom_reporter.plot_image(c_tensor, "yu_eccv20")
    visdom_reporter.plot_image(d_tensor, "zhu_iccp21")

def main():
    # test_lighting()
    #test_deferred_render()
    # compute_and_produce_rgb_v2("azimuth", "0deg", sys.argv)
    # compute_and_produce_rgb_v2("azimuth", "36deg", sys.argv)
    # compute_and_produce_rgb_v2("azimuth", "72deg", sys.argv)
    # compute_and_produce_rgb_v2("azimuth", "108deg", sys.argv)
    # compute_and_produce_rgb_v2("azimuth", "144deg", sys.argv)

    # produce_rgbs("azimuth", "0deg")
    # produce_rgbs("azimuth", "36deg")
    # produce_rgbs("azimuth", "72deg")
    # produce_rgbs("azimuth", "108deg")
    # produce_rgbs("azimuth", "144deg")

    # measure_shading_diff("E:/SynthWeather Dataset 4/azimuth/0deg/shading/", "E:/SynthWeather Dataset 4/azimuth/144deg/shading/")

    measure_performance()


if __name__ == "__main__":
    main()