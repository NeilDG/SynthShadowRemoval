#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --train_mode=train_shadow_matte "
              "--plot_enabled=0 --version=\"v58.27\" --iteration=4")


if __name__ == "__main__":
    main()