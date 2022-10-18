#BASE SCRIPT FOR RUNNING IN GLOUD
import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v55.02\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=3 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v55.03\" --iteration=1")


if __name__ == "__main__":
    main()