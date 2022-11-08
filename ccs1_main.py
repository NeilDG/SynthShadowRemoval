import os

def main():
    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.01\" "
              "--shadow_removal_iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.02\" "
              "--shadow_removal_iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.03\" "
              "--shadow_removal_iteration=1")

    os.system("python \"iid_train_v3.py\" --server_config=2 --cuda_device=\"cuda:2\" --img_to_load=-1 "
              "--plot_enabled=0  --shadow_removal_version=\"v59.04\" "
              "--shadow_removal_iteration=1")

if __name__ == "__main__":
    main()