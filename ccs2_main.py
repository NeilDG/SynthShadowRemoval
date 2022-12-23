import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0  --shadow_matte_network_version=\"v60.15_places\" --shadow_removal_version=\"v60.10_places\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=4")

if __name__ == "__main__":
    main()