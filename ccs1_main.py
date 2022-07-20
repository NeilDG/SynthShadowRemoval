import os

def main():
    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=9")

    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=10")

    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=11")

    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=12")

    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=13")

    os.system("python \"iid_train_v2.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=14")

if __name__ == "__main__":
    main()