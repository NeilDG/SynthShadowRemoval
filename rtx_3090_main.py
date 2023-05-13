#Script to use for running heavy training.

import os

def train_shadow_matte():
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.22_places\" --iteration=1")

    # FOR TESTING
    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=1000 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=1 --save_per_iter=250 --network_version=\"rgb2sm_v61.25_places\" --iteration=1")

def train_shadow_removal():
    os.system("python \"shadow_train_main-3.py\" --server_config=5 --img_to_load=200000 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.26_srd\" --iteration=1")
    #
    # os.system("python \"shadow_train_main-3.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.26_srd\" --iteration=1")

    # os.system("python \"shadow_train_main-2.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_per_iter=500 --network_version=\"rgb2ns_v61.14_places\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.13_places\" --iteration=1")

    #FOR TESTING
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_per_iter=50 --network_version=\"rgb2ns_v61.00_places\" --iteration=1")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=50 --network_version=\"synth2istd_v01.00\" --iteration=1")

def main():
    # train_shadow_removal()
    train_shadow_matte()

    # train_img2img()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
