#Script to use for running heavy training.

import os
def train_shadow_matte():
    os.system("python3 \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.22_istd\" --iteration=1")

    os.system("python3 \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.23_istd\" --iteration=1")

def train_shadow_removal():
    os.system("python3 \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.00_places\" --iteration=1")

def train_img2img():
    os.system("python3 \"train_img2img_main.py\" --server_config=5 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=250 --network_version=\"synth2srd_v01.00\" --iteration=5")

    os.system("python3 \"train_img2img_main.py\" --server_config=5 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=250 --network_version=\"synth2istd_v01.00\" --iteration=5")


def main():
    # train_img2img()
    train_shadow_matte()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()