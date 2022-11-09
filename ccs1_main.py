import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --train_mode=train_shadow_matte "
              "--plot_enabled=0  --shadow_matte_network_version=\"v58.58\" --shadow_removal_version=\"v58.28\" "
              "--shadow_matte_iteration=4 --shadow_removal_iteration=1")

if __name__ == "__main__":
    main()