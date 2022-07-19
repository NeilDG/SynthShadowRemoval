import os

def main():
    os.system("python \"iid_train_v2.py\" --server_config=5 --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=7")

    os.system("python \"iid_train_v2.py\" --server_config=5 --img_to_load=-1 --test_mode=0 "
              "--plot_enabled=0 --version=\"v9.01\" --iteration=8")

if __name__ == "__main__":
    main()