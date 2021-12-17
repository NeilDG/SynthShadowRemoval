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
    LIGHTMAP_PATH = "E:/SynthWeather Dataset 2/lightmap/"

    rgb_list =  dataset_loader.assemble_unpaired_data(RGB_PATH, -1)
    albedo_list = dataset_loader.assemble_unpaired_data(ALBEDO_PATH, -1)
    normal_list = dataset_loader.assemble_unpaired_data(NORMAL_PATH, -1)
    specular_list = dataset_loader.assemble_unpaired_data(SPECULAR_PATH, -1)
    smoothness_list = dataset_loader.assemble_unpaired_data(SMOOTHNESS_PATH, -1)
    lightness_list = dataset_loader.assemble_unpaired_data(LIGHTMAP_PATH, -1)

    for i, (rgb_path, albedo_path, normal_path, specular_path, smoothness_path, lightmap_path)  \
            in enumerate(zip(rgb_list, albedo_list, normal_list, specular_list, smoothness_list, lightness_list)):

        path_segment = rgb_path.split("/")
        file_name = path_segment[len(path_segment) - 1]

        albedo_img = cv2.imread(albedo_path)
        normal_img = cv2.imread(normal_path)
        specular_img = cv2.imread(specular_path)
        smoothness_img = cv2.imread(smoothness_path)
        lightmap_img = cv2.imread(lightmap_path)
        rgb_img = cv2.imread(rgb_path)

        albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        specular_img = cv2.cvtColor(specular_img, cv2.COLOR_BGR2RGB)
        smoothness_img = cv2.cvtColor(smoothness_img, cv2.COLOR_BGR2RGB)
        lightmap_img = cv2.cvtColor(lightmap_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        albedo_img = cv2.normalize(albedo_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
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

        light_color = np.asarray([225, 247, 250]) / 255.0
        # light_color = np.asarray([np.random.randn(), np.random.randn(), np.random.randn()])
        lightmap_img = lightmap_img * light_color - 0.1

        # shading_component = np.clip((rgb_img / albedo_img) - lightmap_img , 0.0, 1.0)
        # rgb_img_like = np.clip((albedo_img * shading_component + lightmap_img), 0.0, 1.0)

        shading_component = np.clip((rgb_img / albedo_img), 0.0, 1.0)
        rgb_img_like = np.clip((albedo_img * shading_component), 0.0, 1.0)

        diff = rgb_img - rgb_img_like
        print("Difference: ", np.mean(diff))

        # plt.imshow(albedo_img)
        # plt.show()

        # plt.imshow(img_mask)
        # plt.show()
        #
        # plt.imshow(shading_component)
        # plt.show()

        # plt.imshow(shading_component[:,:,0], cmap='gray')
        # plt.show()
        #
        # plt.imshow(shading_component[:,:,1], cmap='gray')
        # plt.show()
        #
        # plt.imshow(shading_component[:,:,2], cmap='gray')
        # plt.show()

        # plt.imshow(rgb_img_like)
        # plt.show()
        #
        # plt.imshow(rgb_img)
        # plt.show()
        #
        # break

        cv2.imwrite("E:/SynthWeather Dataset 3/rgb/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(rgb_img_like, alpha=255.0), cv2.COLOR_BGR2RGB))
        cv2.imwrite("E:/SynthWeather Dataset 3/shading/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(shading_component, alpha=255.0), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("E:/SynthWeather Dataset 3/lightmap_img/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(lightmap_img, alpha=255.0), cv2.COLOR_BGR2RGB))
        cv2.imwrite("E:/SynthWeather Dataset 3/albedo/" + file_name, cv2.cvtColor(cv2.convertScaleAbs(albedo_img, alpha=255.0), cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    main()