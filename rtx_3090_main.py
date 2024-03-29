#Script to use for running heavy training.

import os

def train_shadow_matte():
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2sm_v61.68_istd\" --iteration=1")
    #
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2sm_v61.65_istd\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2sm_v61.69_istd\" --iteration=1")

    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2sm_v61.50_istd+srd\" --iteration=1")

    # FOR TESTING
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=1 --save_per_iter=20 --network_version=\"rgb2sm_v61.52_istd+srd\" --iteration=1")

    # #
    # os.system("python \"shadow_train_main-2.py\" --server_config=5 --img_to_load=50 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_every_epoch=20 --epoch_to_load=0 --network_version=\"rgb2ns_v61.test_places\" --iteration=1")

def train_shadow_removal():
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2ns_v61.45_places\" --iteration=1")
    #
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2ns_v61.46_places\" --iteration=1")
    #
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2ns_v61.47_places\" --iteration=1")

    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2ns_v61.48_places\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=10000 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2ns_v61.42_places\" --iteration=1")

    #FOR TESTING
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_per_iter=50 --network_version=\"rgb2ns_v61.43_srd+real\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_per_iter=100 --network_version=\"rgb2ns_v61.40_synshadow\" --iteration=1")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=50 --network_version=\"synth2istd_v01.00\" --iteration=1")

def main():
    # train_shadow_removal()
    train_shadow_matte()
    # train_shadow_removal()

    # train_img2img()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
