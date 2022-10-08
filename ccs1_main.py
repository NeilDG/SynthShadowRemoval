import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v44.01\" --iteration=6")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v44.01\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v44.01\" --iteration=9")

if __name__ == "__main__":
    main()