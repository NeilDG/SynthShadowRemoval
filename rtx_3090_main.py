#Script to use for running heavy training.

import os

def train_shadow_matte():
    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow_matte\" "
              "--plot_enabled=0  --shadow_matte_network_version=\"v60.16_srd\" --shadow_removal_version=\"v60.15_synshadow\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=1")

def train_shadow_removal():
    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0  --shadow_matte_network_version=\"v60.32_places\" --shadow_removal_version=\"v60.33_places\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=1")

def train_domain_adaptation():
    os.system("python \"cyclegan_main.py\" --server_config=5 --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --g_lr=0.0002 --d_lr=0.0002 --version=\"v10.08\" --iteration=5")

def main():
    # train_shadow_matte()
    train_shadow_removal()

    # train_domain_adaptation()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
