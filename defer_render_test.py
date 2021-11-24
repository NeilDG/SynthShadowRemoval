from loaders import dataset_loader
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    RGB_PATH = "E:/SynthWeather Dataset 2/default/"
    ALBEDO_PATH = "E:/SynthWeather Dataset 2/albedo/"
    NORMAL_PATH = "E:/SynthWeather Dataset 2/normal/"
    SPECULAR_PATH = "E:/SynthWeather Dataset 2/specular/"
    SMOOTHNESS_PATH = "E:/SynthWeather Dataset 2/smoothness/"

    rgb_list =  dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)
    normal_list = dataset_loader.assemble_unpaired_data(NORMAL_PATH, -1)
    specular_list = dataset_loader.assemble_unpaired_data(SPECULAR_PATH, -1)
    smoothness_list = dataset_loader.assemble_unpaired_data(SMOOTHNESS_PATH, -1)

    for i, (rgb_path, albedo_path, normal_path, specular_path, smoothness_path) in enumerate(zip(rgb_list, albedo_list, normal_list, specular_list, smoothness_list)):
        albedo_img = cv2.imread(albedo_path)
        normal_img = cv2.imread(normal_path)
        specular_img = cv2.imread(specular_path)
        smoothness_img = cv2.imread(smoothness_path)
        rgb_img = cv2.imread(rgb_path)

        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        specular_img = cv2.cvtColor(specular_img, cv2.COLOR_BGR2RGB)
        smoothness_img = cv2.cvtColor(smoothness_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normal_img = cv2.normalize(normal_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        specular_img = cv2.normalize(specular_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        smoothness_img = cv2.normalize(smoothness_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # diffuse_contribution = albedo_img * normal_img * smoothness_img * 0.45
        # specular_highlights = (0.45 * specular_img * normal_img * smoothness_img)
        composited_img = albedo_img  * ((albedo_img * specular_img) + (albedo_img * smoothness_img))

        plt.imshow(albedo_img)
        plt.show()

        plt.imshow(normal_img)
        plt.show()

        plt.imshow(composited_img)
        plt.show()

        plt.imshow(rgb_img)
        plt.show()

        break

if __name__ == "__main__":
    main()