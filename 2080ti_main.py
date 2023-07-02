#Script to use for running heavy training.

import os
def train_shadow_matte():
    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.70_istd\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.71_istd\" --iteration=1")
    #
    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.72_istd\" --iteration=1")

    os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2sm_v61.73_istd\" --iteration=1")


def train_shadow_removal():
    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.43_istd+real\" --iteration=1")
    #
    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.43_srd+real\" --iteration=1")

    # os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.44_istd+real\" --iteration=1")

    os.system("python \"shadow_train_main.py\" --server_config=2 --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2ns_v61.44_srd+real\" --iteration=1")

def train_domain_adaptation():
    os.system("python \"cyclegan_main.py\" --server_config=5 --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v10.08\" --iteration=5")


def main():
    # train_shadow_removal()
    train_shadow_matte()
    # train_domain_adaptation()
    os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()