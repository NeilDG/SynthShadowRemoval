#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def main():
    os.system("python \"cyclegan_main.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=1 --g_lr=0.0002 --d_lr=0.0002 --version=\"v9.04\" --iteration=2")

    os.system("python \"cyclegan_main.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=1 --g_lr=0.0002 --d_lr=0.0002 --version=\"v9.05\" --iteration=2")


if __name__ == "__main__":
    main()