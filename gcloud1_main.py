#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --train_mode=train_shadow "
              "--plot_enabled=0 --shadow_matte_network_version=\"v58.34\" --shadow_removal_version=\"v58.34\" "
              "--shadow_matte_iteration=1 --shadow_removal_iteration=1")



if __name__ == "__main__":
    main()