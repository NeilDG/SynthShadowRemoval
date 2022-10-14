import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v54.02\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v54.03\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v54.04\" --iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:1\" --img_to_load=-1 --debug_run=0 "
              "--plot_enabled=0 --version=\"v54.05\" --iteration=1")

if __name__ == "__main__":
    main()