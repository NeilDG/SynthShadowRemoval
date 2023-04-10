from loaders import dataset_loader
import global_config
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    PATH_A = constants.DATASET_WEATHER_STYLED_PATH
    PATH_B = constants.DATASET_WEATHER_SEGMENT_PATH

    img_list = dataset_loader.assemble_unpaired_data(PATH_A, -1)

    for i in range(len(img_list)):
        img_a_name = img_list[i]
        path_segment = img_a_name.split("/")
        file_name = path_segment[len(path_segment) - 1]

        img_b_name = PATH_B + file_name

        img_a = cv2.cvtColor(cv2.imread(img_a_name), cv2.COLOR_BGR2RGB)
        img_segment = cv2.cvtColor(cv2.imread(img_b_name), cv2.COLOR_BGR2RGB)
        img_segment = cv2.resize(img_segment, (256, 256))
        
        #convert img_b to mask
        img_b = cv2.inRange(img_segment[:,:,1], 200, 255) #green segmentation mask = road
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)

        img_c = cv2.inRange(img_segment[:, :, 0], 200, 255)  # red segmentation mask = building
        img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)

        plt.imshow(img_a)
        plt.show()

        print(np.shape(img_a), np.shape(img_b))

        plt.imshow(cv2.bitwise_and(img_a, img_b))
        plt.show()

        plt.imshow(cv2.bitwise_and(img_a, img_c))
        plt.show()

if __name__=="__main__":
    main()