import os

def main():
    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=5 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=6 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=7 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=8 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=9 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=10 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=11 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

    os.system("python \"iid_train.py\" --server_config=2 --cuda_device=\"cuda:0\" --img_to_load=-1 --load_previous=0 --test_mode=0 --patch_size=64 --batch_size=256 --min_epochs=20 "
              "--net_config=2 --num_blocks=0 "
              "--plot_enabled=1 --debug_mode=0 --version_name=\"iid_networkv6.01\" --iteration=12 "
              "--da_enabled=1 --da_version_name=\"embedding_v5.00_5\"")

if __name__ == "__main__":
    main()