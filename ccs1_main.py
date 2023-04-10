import os

def main():
    os.system("python \"shadow_train_main.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0  --shadow_matte_network_version=\"v60.24_places\" --shadow_removal_version=\"v60.21_places\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=1")

    os.system("python \"shadow_train_main.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0  --shadow_matte_network_version=\"v60.24_places\" --shadow_removal_version=\"v60.21_places\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=4")

if __name__ == "__main__":
    main()