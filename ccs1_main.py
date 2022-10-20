import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v50.07\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v50.07\" --iteration=2")

if __name__ == "__main__":
    main()