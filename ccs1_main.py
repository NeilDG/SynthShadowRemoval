import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.05\" "
              "--shadow_removal_iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.06\" "
              "--shadow_removal_iteration=1")

if __name__ == "__main__":
    main()